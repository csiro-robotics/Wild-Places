# Adapted from MinkLoc3Dv2
# by Ethan Griffiths (Data61, Pullenvale)
# Train MinkLoc model

import os
import numpy as np
import torch
import tqdm
import pathlib
import typing as tp
import wandb
import submitit
from timm.utils.model_ema import ModelEmaV3
from timm.optim.lamb import Lamb

from misc.torch_utils import to_device
from misc.utils import TrainingParams, get_datetime, set_seed, update_params_from_dict
from models.losses.loss import make_losses, kdloss
from models.model_factory import model_factory
from datasets.dataset_utils import make_dataloaders
from eval.pnv_evaluate import evaluate, print_eval_stats, pnv_write_eval_stats

class NetworkTrainer:
    """
    Class to handle main training loop for HOTFormerLoc. Supports checkpointing
    and automatic resubmission to SLURM clusters through submitit.
    """

    def __init__(self):
        self.model = None
        self.model_ema = None
        self.optimizer = None
        self.scheduler = None
        self.params = None
        self.model_pathname = None
        self.device = None
        self.wandb_id = None
        self.resume = False
        self.start_epoch = 1
        self.curr_epoch = 1
        self.best_avg_AR_1 = 0.0
        self.checkpoint_extension = '_latest.ckpt'
        
    def __call__(
        self,
        params: TrainingParams = None,
        *args,
        **kwargs,
    ):
        """
        Handle logic for starting/resuming training when called manually or by
        submitit.
        """
        checkpoint_path = kwargs.get('checkpoint_path')
        self.resume = checkpoint_path is not None        
        # Set params for hyperparam search
        self.params = params
        if not self.resume:
            if self.params.hyperparam_search:
                if len(args) == 1 and isinstance(args[0], dict):  # This is required for submitit job arrays currently
                    kwargs = args[0]
                assert kwargs != {}, 'No valid hyperparams were provided for search'
                self.params = update_params_from_dict(self.params, kwargs)
        self.params.print()
        
        # Seed RNG
        set_seed()
        print('Determinism: Enabled')

        # Initialise model, optimiser, and scheduler 
        self.init_model_optim_sched()
        self.save_dir = kwargs.get('save_dir')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Load state from ckpt if resubmitted, otherwise start from scratch
        if self.resume:
            self.model_pathname = os.path.join(self.save_dir, "model")
            # self.model_pathname = checkpoint_path.split(self.checkpoint_extension)[0]
            state = torch.load(checkpoint_path)
            try:
                self.start_epoch = state['epoch']
                self.curr_epoch = self.start_epoch
                self.wandb_id = state['wandb_id']
                self.best_avg_AR_1 = state['best_avg_AR_1']
                self.model.load_state_dict(state['model_state_dict'])
                self.optimizer.load_state_dict(state['optim_state_dict'])
                if self.scheduler is not None:
                    self.scheduler.load_state_dict(state['sched_state_dict'])
                if self.model_ema is not None:
                    self.model_ema.load_state_dict(state['model_ema_state_dict'])
            except KeyError:
                print("Invalid checkpoint file provided. Only files ending in '.ckpt' are valid for resuming training.")
            print(f'Resuming training of {self.model_pathname} from epoch {self.start_epoch}')
        else:
            # Create model class
            s = get_datetime()
            model_name = self.params.model_params.model + '_' + s
            # Add SLURM job ID to prevent overwriting paths for jobs running at same time
            if 'SLURM_JOB_ID' in os.environ:
                model_name += f"_job{os.environ['SLURM_JOB_ID']}"
            weights_path = self.create_weights_folder(self.params.dataset_name)
            self.model_pathname = os.path.join(self.save_dir, "model")
            print('Model name: {}'.format(model_name))
            
        if hasattr(self.model, 'print_info'):
            self.model.print_info()
        else:
            n_params = sum([param.nelement() for param in self.model.parameters()])
            print('Number of model parameters: {}'.format(n_params))
                    
        # Begin train loop
        self.do_train()

    def checkpoint(self, *args: tp.Any, **kwargs: tp.Any) -> submitit.helpers.DelayedSubmission:
        """
        This function is called asynchronously by submitit when a job is timed
        out. Dumps the current model state to disk and returns a
        DelayedSubmission to resubmit the job.
        """
        checkpoint_path = self.model_pathname + self.checkpoint_extension
        print(f'Training interupted at epoch {self.curr_epoch}. '
              f'Saving ckpt to {checkpoint_path} and resubmitting.')
        if not os.path.exists(checkpoint_path):
            self.save_checkpoint(checkpoint_path)
        training_callable = NetworkTrainer()
        delayed_submission = submitit.helpers.DelayedSubmission(
            training_callable,
            self.params,
            checkpoint_path=checkpoint_path,
        )
        return delayed_submission

    def save_checkpoint(self, checkpoint_path: str):
        # Save checkpoint of training state (as opposed to just model weights)
        print(f"[INFO] Saving checkpoint to {checkpoint_path}", flush=True)
        state = {
            'epoch': self.curr_epoch,
            'wandb_id': self.wandb_id,
            'best_avg_AR_1': self.best_avg_AR_1,
            'model_state_dict': self.model.state_dict(),                
            'optim_state_dict': self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            state['sched_state_dict'] = self.scheduler.state_dict()
        if self.model_ema is not None:
            state['model_ema_state_dict'] = self.model_ema.state_dict()
        torch.save(state, checkpoint_path)

    def init_model_optim_sched(self):
        """
        Initialise the model, optimiser, and scheduler from the provided parameters.
        """
        self.model = model_factory(self.params.model_params)

        # Move the model to the proper device before configuring the optimizer
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.model.to(self.device)
        print('Model device: {}'.format(self.device))

        # Setup exponential moving average of model weights, SWA could be used here too
        if self.params.mesa > 0.0:
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
            self.model_ema = ModelEmaV3(self.model, decay=0.9998)

        # Training elements
        if self.params.optimizer == 'Adam':
            optimizer_fn = torch.optim.Adam
        elif self.params.optimizer == 'AdamW':
            optimizer_fn = torch.optim.AdamW
        elif self.params.optimizer == 'Lamb':
            optimizer_fn = Lamb
        else:
            raise NotImplementedError(f"Unsupported optimizer: {self.params.optimizer}")

        if self.params.weight_decay is None or self.params.weight_decay == 0:
            self.optimizer = optimizer_fn(self.model.parameters(), lr=self.params.lr)
        else:
            self.optimizer = optimizer_fn(self.model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay)

        if self.params.scheduler is not None:
            if self.params.scheduler == 'CosineAnnealingLR':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.params.epochs+1,
                                                                    eta_min=self.params.min_lr)
            elif self.params.scheduler == 'MultiStepLR':
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.params.scheduler_milestones, gamma=self.params.gamma)
            elif self.params.scheduler == 'ExponentialLR':
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.params.gamma)
            else:
                raise NotImplementedError('Unsupported LR scheduler: {}'.format(self.params.scheduler))

        if self.params.warmup_epochs is not None:
            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.warmup)
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, [warmup_scheduler, self.scheduler], [self.params.warmup_epochs])

    def warmup(self, epoch: int):
        # Linear scaling lr warmup
        min_factor = 1e-3
        return max(float(epoch / self.params.warmup_epochs), min_factor)

    def tensors_to_numbers(self, stats):
        stats = {e: stats[e].item() if torch.is_tensor(stats[e]) else stats[e] for e in stats}
        return stats

    def print_global_stats(self, phase, stats):
        s = f"{phase}  loss: {stats['loss']:.4f}   embedding norm: {stats['avg_embedding_norm']:.3f}  "
        if 'num_triplets' in stats:
            s += f"Triplets (all/active): {stats['num_triplets']:.1f}/{stats['num_non_zero_triplets']:.1f}  " \
                f"Mean dist (pos/neg): {stats['mean_pos_pair_dist']:.3f}/{stats['mean_neg_pair_dist']:.3f}   "
        if 'positives_per_query' in stats:
            s += f"#positives per query: {stats['positives_per_query']:.1f}   "
        if 'best_positive_ranking' in stats:
            s += f"best positive rank: {stats['best_positive_ranking']:.1f}   "
        if 'recall' in stats:
            s += f"Recall@1: {stats['recall'][1]:.4f}   "
        if 'ap' in stats:
            s += f"AP: {stats['ap']:.4f}   "

        print(s, flush=True)

    def print_stats(self, phase, stats):
        self.print_global_stats(phase, stats['global'])
    
    def create_weights_folder(self, dataset_name : str):
        # Create a folder to save weights of trained models
        this_file_path = pathlib.Path(__file__).parent.absolute()
        temp, _ = os.path.split(this_file_path)
        weights_path = os.path.join(temp, 'weights', dataset_name)
        if not os.path.exists(weights_path):
            os.makedirs(weights_path)
        assert os.path.exists(weights_path), 'Cannot create weights folder: {}'.format(weights_path)
        return weights_path

    def log_eval_stats(self, stats):
        eval_stats = {}
        for database_name in stats:
            eval_stats[database_name] = {}
            eval_stats[database_name]['recall@1%'] = stats[database_name]['ave_one_percent_recall']
            eval_stats[database_name]['recall@1'] = stats[database_name]['ave_recall'][0]
            eval_stats[database_name]['MRR'] = stats[database_name]['ave_mrr']
        return eval_stats

    def training_step(self, global_iter, phase, loss_fn, mesa=0.0):
        assert phase in ['train', 'val']

        batch, positives_mask, negatives_mask = next(global_iter)

        if self.params.verbose:
            print("[INFO] Batch loaded, begin forward pass", flush=True)

        batch = to_device(batch, self.device, non_blocking=True, construct_octree_neigh=True)

        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        self.optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            y = self.model(batch)
            stats = self.model.stats.copy() if hasattr(self.model, 'stats') else {}

            embeddings = y['global']

            loss, temp_stats = loss_fn(embeddings, positives_mask, negatives_mask)
            temp_stats = self.tensors_to_numbers(temp_stats)
            stats.update(temp_stats)
            if phase == 'train':
                # Compute MESA loss
                if mesa > 0.0:
                    with torch.no_grad():
                        ema_output = self.model_ema.module(batch)['global'].detach()
                    kd = kdloss(embeddings, ema_output)
                    loss += mesa * kd
                    
                loss.backward()
                self.optimizer.step()
                
                if self.model_ema is not None:
                    self.model_ema.update(self.model)

        torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

        return stats


    def multistaged_training_step(self, global_iter, phase, loss_fn, mesa=0.0):
        # Training step using multistaged backpropagation algorithm as per:
        # "Learning with Average Precision: Training Image Retrieval with a Listwise Loss"
        # This method will break when the model contains Dropout, as the same mini-batch will produce different embeddings.
        # Make sure mini-batches in step 1 and step 3 are the same (so that BatchNorm produces the same results)
        # See some exemplary implementation here: https://gist.github.com/ByungSun12/ad964a08eba6a7d103dab8588c9a3774

        assert phase in ['train', 'val']
        batch, positives_mask, negatives_mask = next(global_iter)
        
        if self.params.verbose:
            print("[INFO] Batch loaded, begin forward pass", flush=True)

        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        # Stage 1 - calculate descriptors of each batch element (with gradient turned off)
        # In training phase network is in the train mode to update BatchNorm stats
        embeddings_l = []
        embeddings_ema_l = []
        with torch.set_grad_enabled(False):
            for minibatch in batch:
                minibatch = to_device(minibatch, self.device, non_blocking=True, construct_octree_neigh=True)
                y = self.model(minibatch)
                embeddings_l.append(y['global'])            
                # Compute MESA embeddings
                if mesa > 0.0:
                    ema_output = self.model_ema.module(minibatch)['global'].detach()
                    embeddings_ema_l.append(ema_output)

        torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

        # Stage 2 - compute gradient of the loss w.r.t embeddings
        embeddings = torch.cat(embeddings_l, dim=0)
        if mesa > 0.0:
            embeddings_ema = torch.cat(embeddings_ema_l, dim=0)

        with torch.set_grad_enabled(phase == 'train'):
            if phase == 'train':
                embeddings.requires_grad_(True)
            loss, stats = loss_fn(embeddings, positives_mask, negatives_mask)
            stats = self.tensors_to_numbers(stats)
            # Compute MESA loss
            if mesa > 0.0:
                kd = kdloss(embeddings, embeddings_ema)
                loss += mesa * kd
            if phase == 'train':
                loss.backward()
                embeddings_grad = embeddings.grad

        # Delete intermediary values
        embeddings_l, embeddings, embeddings_ema_l, embeddings_ema, y, loss = [None]*6

        # Stage 3 - recompute descriptors with gradient enabled and compute the gradient of the loss w.r.t.
        # network parameters using cached gradient of the loss w.r.t embeddings
        if phase == 'train':
            self.optimizer.zero_grad()
            i = 0
            with torch.set_grad_enabled(True):
                for minibatch in batch:
                    minibatch = to_device(minibatch, self.device, non_blocking=True, construct_octree_neigh=True)
                    y = self.model(minibatch)
                    embeddings = y['global']
                    minibatch_size = len(embeddings)
                    # Compute gradients of network params w.r.t. the loss using the chain rule (using the
                    # gradient of the loss w.r.t. embeddings stored in embeddings_grad)
                    # By default gradients are accumulated
                    embeddings.backward(gradient=embeddings_grad[i: i+minibatch_size])
                    i += minibatch_size

                self.optimizer.step()
                
                if self.model_ema is not None:
                    self.model_ema.update(self.model)

        torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors
        return stats

    def do_train(self):        
        # set up dataloaders
        dataloaders = make_dataloaders(self.params, validation=self.params.validation)

        loss_fn = make_losses(self.params)

        if self.params.batch_split_size is None or self.params.batch_split_size == 0:
            train_step_fn = self.training_step
        else:
            # Multi-staged training approach with large batch split into multiple smaller chunks with batch_split_size elems
            train_step_fn = self.multistaged_training_step

        ###########################################################################
        # Initialize Weights&Biases logging service
        ###########################################################################

        params_dict = {e: self.params.__dict__[e] for e in self.params.__dict__ if e != 'model_params'}
        model_params_dict = {"model_params." + e: self.params.model_params.__dict__[e] for e in self.params.model_params.__dict__}
        params_dict.update(model_params_dict)
        n_params = sum([param.nelement() for param in self.model.parameters()])
        params_dict['num_params'] = n_params
        if self.params.wandb:
            wandb.init(project='HOTFormerLoc', config=params_dict, id=self.wandb_id, resume="allow", dir=self.save_dir)
            self.wandb_id = wandb.run.id
        ###########################################################################
        #
        ###########################################################################

        # Training statistics
        stats = {'train': [], 'eval': []}

        if 'val' in dataloaders:
            # Validation phase
            phases = ['train', 'val']
            stats['val'] = []
        else:
            phases = ['train']

        for epoch in tqdm.tqdm(range(self.start_epoch, self.params.epochs + 1),
                               initial=self.start_epoch-1, total=self.params.epochs):
            metrics = {'train': {}, 'val': {}, 'test': {}}      # Metrics for wandb reporting
            if epoch / self.params.epochs > self.params.mesa_start_ratio:
                mesa = self.params.mesa
            else:
                mesa = 0.0
            
            for phase in phases:
                if self.params.verbose:
                    print(f"[INFO] Begin {phase} phase", flush=True)
                running_stats = []  # running stats for the current epoch and phase
                count_batches = 0

                if phase == 'train':
                    global_iter = iter(dataloaders['train'])
                else:
                    global_iter = None if dataloaders['val'] is None else iter(dataloaders['val'])

                while True:
                    count_batches += 1
                    batch_stats = {}
                    if self.params.debug and count_batches > 2:
                        break
                    if self.params.verbose:
                        print(f"[INFO] Processing {phase} batch: {count_batches}",
                            flush=True)

                    try:
                        temp_stats = train_step_fn(global_iter, phase, loss_fn, mesa)
                        batch_stats['global'] = temp_stats

                    except StopIteration:
                        # Terminate the epoch when one of dataloders is exhausted
                        break

                    running_stats.append(batch_stats)
                    if phase == 'train':
                        log_dict = {}
                        log_keys = [k for k in running_stats[0]['global'].keys() if isinstance(running_stats[0]['global'][k], float)]
                        for k in log_keys:
                            log_dict["STEP/" + k] = np.mean([x['global'][k] for x in running_stats])
                        #     print(k, round(running_stats[-1]['global'][k], 2), ))
                        # for k in log_keys:
                        #     log_dict[f"STEP/{k}"] = np.mean([x['global'][k] for x in running_stats if isinstance(x, float)])
                        wandb.log(log_dict)

                # Compute mean stats for the phase
                epoch_stats = {}
                for substep in running_stats[0]:
                    epoch_stats[substep] = {}
                    for key in running_stats[0][substep]:
                        temp = [e[substep][key] for e in running_stats]
                        if type(temp[0]) is dict:
                            epoch_stats[substep][key] = {key: np.mean([e[key] for e in temp]) for key in temp[0]}
                        elif type(temp[0]) is np.ndarray:
                            # Mean value per vector element
                            epoch_stats[substep][key] = np.mean(np.stack(temp), axis=0)
                        else:
                            epoch_stats[substep][key] = np.mean(temp)

                stats[phase].append(epoch_stats)
                self.print_stats(phase, epoch_stats)

                # Log metrics for wandb
                metrics[phase]['loss1'] = epoch_stats['global']['loss']
                if 'num_non_zero_triplets' in epoch_stats['global']:
                    metrics[phase]['active_triplets1'] = epoch_stats['global']['num_non_zero_triplets']

                if 'positive_ranking' in epoch_stats['global']:
                    metrics[phase]['positive_ranking'] = epoch_stats['global']['positive_ranking']

                if 'recall' in epoch_stats['global']:
                    metrics[phase]['recall@1'] = epoch_stats['global']['recall'][1]

                if 'ap' in epoch_stats['global']:
                    metrics[phase]['AP'] = epoch_stats['global']['ap']
                    
                

            # ******* FINALIZE THE EPOCH *******
            self.curr_epoch += 1  # increment the epoch counter here to ensure next ckpt saves with correct epoch count
            if self.scheduler is not None:
                self.scheduler.step()
            
            if not self.params.debug:
                checkpoint_path = self.model_pathname + self.checkpoint_extension
                self.save_checkpoint(checkpoint_path)
                if self.params.save_freq > 0 and epoch % self.params.save_freq == 0:
                    epoch_pathname = f"{self.model_pathname}_e{epoch}.ckpt"
                    self.save_checkpoint(epoch_pathname)

            if self.params.eval_freq > 0 and epoch % self.params.eval_freq == 0:
                if self.params.verbose:
                    print("[INFO] Begin evaluation", flush=True)
                eval_stats = evaluate(self.model, self.device, self.params,
                                      log=False, show_progress=self.params.verbose)
                print_eval_stats(eval_stats)
                metrics['test'] = self.log_eval_stats(eval_stats)
                # store best AR@1 on all test sets
                avg_AR_1 = metrics['test']['average']['recall@1']
                if avg_AR_1 > self.best_avg_AR_1:
                    print(f"New best avg AR@1 at Epoch {epoch}: {self.best_avg_AR_1:.2f} -> {avg_AR_1:.2f}")
                    self.best_avg_AR_1 = avg_AR_1
                    if not self.params.debug:
                        best_model_pathname = f"{self.model_pathname}_best.ckpt"
                        self.save_checkpoint(best_model_pathname)

            if not self.params.debug:
                wandb.log(metrics)
                # trigger_sync()

            if self.params.batch_expansion_th is not None:
                # Dynamic batch size expansion based on number of non-zero triplets
                # Ratio of non-zero triplets
                le_train_stats = stats['train'][-1]  # Last epoch training stats
                rnz = le_train_stats['global']['num_non_zero_triplets'] / le_train_stats['global']['num_triplets']
                if rnz < self.params.batch_expansion_th:
                    dataloaders['train'].batch_sampler.expand_batch()

        print('')

        # Save final model weights
        if not self.params.debug:
            final_model_path = self.model_pathname + '_final.ckpt'
            self.save_checkpoint(final_model_path)

        # Evaluate the final
        # PointNetVLAD datasets evaluation protocol
        stats = evaluate(self.model, self.device, self.params, log=False,
                         show_progress=self.params.verbose)
        print_eval_stats(stats)

        print('.')

        # Append key experimental metrics to experiment summary file
        if not self.params.debug:
            model_params_name = os.path.split(self.params.model_params.model_params_path)[1]
            config_name = os.path.split(self.params.params_path)[1]
            model_name = os.path.splitext(os.path.split(final_model_path)[1])[0]
            prefix = "{}, {}, {}".format(model_params_name, config_name, model_name)

            pnv_write_eval_stats(f"pnv_{self.params.dataset_name}_results.txt", prefix, stats)        

        # Return optimization value (to minimize)
        return (1 - self.best_avg_AR_1/100.0)
