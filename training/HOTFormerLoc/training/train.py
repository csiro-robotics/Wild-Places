# Warsaw University of Technology
# Train MinkLoc model

import argparse
import torch

from training.trainer import NetworkTrainer
from misc.utils import TrainingParams


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train HOTFormerLoc model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True,
                        help='Path to the model-specific configuration file')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume from the given checkpoint. Ensure config and model_config matches the supplied checkpoint.')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.set_defaults(debug=False)
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.set_defaults(verbose=False)

    args = parser.parse_args()
    print('Training config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    if args.resume_from is not None:
        print('Resuming from checkpoint path: {}'.format(args.resume_from))
    print('Debug mode: {}'.format(args.debug))
    print('Verbose mode: {}'.format(args.verbose))
    
    params = TrainingParams(args.config, args.model_config, debug=args.debug,
                            verbose=args.verbose)
    # params.print()

    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    training_callable = NetworkTrainer()
    training_callable(params, save_dir=args.save_dir, checkpoint_path=args.resume_from)
