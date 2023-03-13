<html>
   <head></head>
   <body>
      <link rel="stylesheet" type="text/css" href="./github.css" id="_theme">
      <div id="_html" class="markdown-body">
         <meta charset="UTF-8">
         <meta name="description" content="Wild-Places: A Large-Scale Dataset for Lidar Place Recognition in Unstructured Natural Environments">
         <meta name="keywords" content="wild-places dataset, offroad, natural, 3d, place recognition">
         <link rel="shortcut icon" href="./favicon.ico">
         <div align="center">
            <!-- <h1 id="Wild-Places: A Large-Scale Dataset for Lidar Place Recognition in Unstructured Natural Environments">Wild-Places: A Large-Scale Dataset for Lidar Place Recognition in Unstructured Natural Environments</h1> -->
            <p><strong>
               <a href="https://scholar.google.com/citations?user=RxbGr2EAAAAJ&hl=en">Joshua Knights<sup>*,1,2</sup></a> &nbsp;&nbsp;&nbsp;
               <a href="https://scholar.google.com/citations?user=BUvScBEAAAAJ&hl=en">Kavisha Vidanapathirana<sup>*,1,2</sup></a> &nbsp;&nbsp;&nbsp;
               <a href="https://scholar.google.com/citations?hl=en&user=fn-lMpMAAAAJ">Milad Ramezani<sup>1</sup></a> &nbsp;&nbsp;&nbsp;
               <br><a href="https://scholar.google.com/citations?user=v8-lMdUAAAAJ&hl=en">Sridha Sridharan<sup>2</sup></a> &nbsp;&nbsp;&nbsp;
               <a href="https://scholar.google.com.au/citations?user=VpaJsNQAAAAJ&hl=en">Clinton Fookes<sup>2</sup></a> &nbsp;&nbsp;&nbsp;
               <a href="https://scholar.google.com.au/citations?user=QAVcuWUAAAAJ&hl=en">Peyman Moghadam<sup>1,2</sup></a> &nbsp;&nbsp;&nbsp; </strong></p>
               <p> Proceeding of ICRA2023
               <p><sup>*</sup> Equal Contribution
               <br><sup>1</sup>Robotics and Autonomous Systems Group, DATA61, CSIRO, Australia.  E-mails: <span style="font-family:courier;"> firstname.lastname@data61.csiro.au</span>
               <br><sup>2</sup>School of Electrical Engineering and Robotics, Queensland University of Technology (QUT), Australia.  E-mails: <span style="font-family:courier;"> {s.sridharan,c.fookes}@qut.edu.au</span>
            </p>
         </div>
            <hr>
         <div align="center">
            <h3 id="abstract--paper--SCD--Proposed-baseline-method-on-SCD--leaderboard--attn">
            <a href="#1-abstract">Abstract</a> | 
            <a href="#2-paper">Paper</a> | 
            <a href="#3-Citation">Citation</a> |
            <a href="#4-GitHub">GitHub</a> |
            <a href="#5-Images">Images</a> | 
            <a href="#6-Dataset">Dataset Information</a> | 
            <a href="#7-Benchmarking">Benchmarking</a> | 
            <a href="#8-Download">Download</a></h3>
         </div>
         <!-- 
            ###################
            SECTION 1 - ABSTRACT
            ###################
          -->
         <h2 id="1-abstract"><a class="anchor" name="1-abstract" href="#1-abstract"><span class="octicon octicon-link"></span></a>1. Abstract</h2>
         <p style="text-align: justify"><em>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Many existing datasets for lidar place recognition are solely representative of structured urban environments, and have recently been saturated in performance by deep learning based approaches. Natural and unstructured environments present many additional challenges for the tasks of long-term localisation but these environments are not represented in currently available datasets.  To address this we introduce Wild-Places, a challenging large-scale dataset for lidar place recognition in unstructured, natural environments.  Wild-Places contains eight lidar sequences collected with a handheld sensor payload over the course of fourteen months, containing a total of 63K undistorted lidar submaps along with accurate 6DoF ground truth.  Our dataset contains multiple revisits both within and between sequences, allowing for both intra-sequence (i.e. loop closure detection) and inter-sequence (i.e. re-localisation) place recognition.  We also benchmark several state-of-the-art approaches to demonstrate the challenges that this dataset introduces, particularly the case of long-term place recognition due to natural environments changing over time. </em></p>
         <!-- 
            ###################
            SECTION 2 - PAPER
            ###################
          -->
         <h2 id="2-paper"><a class="anchor" name="2-paper" href="#2-paper"><span class="octicon octicon-link"></span></a>2. Paper</h2>
         <div align="center">
            <a href="https://arxiv.org/abs/2211.12732">
            <img  src="images/frontpage.png" height="440" width="340" />
            </a>   
            <p><a href="https://arxiv.org/abs/2211.12732"><strong>Pre-print of accepted ICRA2023 paper: "Wild-Places: A Large-Scale Dataset for Lidar Place Recognition in Unstructured Natural Environments"</strong></a></p>
         </div>
         
          <!-- 
            ###################
            SECTION 3 Citation
            ###################
          -->
          
          <h2 id="3-Citation"><a class="anchor" name="3-paper" href="#3-paper"><span class="octicon octicon-link"></span></a>3. Citation</h2>
          <p>
            If you find this paper helpful for your research, please cite our paper using the following reference: <br><br>
            
         
            @inproceedings{2023wildplaces, <br>
                 title={Wild-Places: A Large-Scale Dataset for Lidar Place Recognition in Unstructured Natural Environments},<br>
                 author={Knights, Joshua and Vidanapathirana, Kavisha and Ramezani, Milad and Sridharan, Sridha and Fookes, Clinton and Moghadam, Peyman},<br>
                 year={2023},<br>
                 booktitle={IEEE International Conference on Robotics and Automation (ICRA)},<br>
                 eprint={arXiv preprint arXiv:2211.12732}<br>
            }
            
          </p>
          
         <!-- 
            ###################
            SECTION 4 Github
            ###################
          -->
          <h2 id="4-GitHub"><a class="anchor" name="4-paper" href="#4-paper"><span class="octicon octicon-link"></span></a>4. GitHub</h2>
          <p>
            We provide training and evaluation code for our dataset <a href="https://github.com/csiro-robotics/Wild-Places/tree/master">in our GitHub repository</a> as well as example script for loading the dataset and forming the training and testing splits outlined in our paper.
          </p>
         <!-- 
            ###################
            SECTION 5 - IMAGES
            ###################
          -->
         <h2 id="5-Images"><a class="anchor" name="5-Wild-Places" href="#5-Images"><span class="octicon octicon-link"></span></a>5. Images</h2>
         <h4 id="51-environments"><a class="anchor" name="51-environments" href="#51-environments"><span class="octicon octicon-link">
         </span></a>5.1 Sequences & Environments</h4>
         <div align="center">
            <img  src="images/combined_env_img.png" height="600"/>
         </div>
         <p>Environment and sequence visualisation.  The top row shows the trajectory of two sequences - V-03 and K-02 - overlaid on a satellite image of their respective environments.  The bottom row shows the trajectory of sequences 01, 02, 03 & 04 on each environment from top to bottom, respectively.</p>
         <h4 id="52-Global-Map"><a class="anchor" name="52-Global-Map" href="#52-Global-Map"><span class="octicon octicon-link"></span>
         </a>5.2 Global Map</h4>
         <div align="center">
            <img src="images/venman12month_fly.gif" height="500"/>
         </div>
         <p> Visualisation of the Global Map.  We use Wildcat SLAM to create an undistorted global point cloud map for each sequence, from which we extract the submaps used during training and evaluation.</p>
         <h4 id="53-Diversity"><a class="anchor" name="53-Diversity" href="#53-Diversity"><span class="octicon octicon-link">
         </span></a>5.3 Diversity</h4>
         <div align="center">
            <img src="images/diversity.png" height="440" />
         </div>
         <p>Visualisation of the diversity in environments present in the Wild-Places dataset.  The top row shows RGB images from the front camera of the sensor payload, while the bottom row shows the point cloud visualisation corresponding to the camera pose in the global point cloud.</p>
        <!-- 
            ###################
            SECTION 6 - Dataset Information
            ###################
          -->
        <h2 id="6-Dataset"><a class="anchor" name="6-Wild-Places" href="#6-DATASET"><span class="octicon octicon-link">
        </span></a>6. Dataset Information</h2>
        <h4 id="61-Comparison"><a class="anchor" name="61-Comparison" href="#61-Comparison"><span class="octicon octicon-link">
         </span></a>6.1 Comparison</h4>
         <div align="center">
            <img  src="images/comparison_datasets_table.png" height="250"/>
         </div>
         <p>Comparison of public lidar datasets. The top half of the table shows the most popular lidar datasets used for large-scale localisation evaluation. The bottom half shows public lidar datasets which contain only natural and unstructured environments. Wild-Places is the only dataset that satisfies both of these criteria.  We define long-term revisits here as a time gap greater than 1 year. <sup>*</sup> Post-processed variation introduced in PointNetVLAD</p>
         <h4 id="62-Sequences"><a class="anchor" name="62-Sequences" href="#62-Sequences"><span class="octicon octicon-link">
         </span></a>6.2 Sequences</h4>
         <div align="center">
            <img  src="images/sequences_table.png" height="200"/>
         </div>
         Per-Sequence Information for the Wild-Places Dataset.  We collect a total of eight lidar sequences in two environments over the span of fourteen months for a total of 35.33km traversed distance and approximately 67K point cloud submaps.
         <!-- 
            ###################
            SECTION 7 - Benchmarking
            ###################
          -->
        <h2 id="7-Benchmarking"><a class="anchor" name="7-Benchmarking" href="#7-Benchmarking"><span class="octicon octicon-link"></span></a>7. Benchmarking</h2>
        <!-- <h4 id="71-Table"><a class="anchor" name="71-Table" href="#71-Table"><span class="octicon octicon-link">
        </span></a>7.1 Benchmarking Results</h4> -->
        <div align="center">
            <img  src="images/results.png" height="200"/>
         </div>
         <div align="center">
            <img  src="images/performance_drop.png" height="350"/>
         </div>
         We benchmark four state-of-the-art approaches on our dataset: ScanContext as a strong handcrafted baseline, and TransLoc3D, MinkLoc3Dv2 and LoGG3D-Net as learning-based approaches.  For both intra and inter-run place recognition our dataset challenges the existing state-of-art.  In addition, in the inter-run place recognition task we demonstrate the challenge posed by long-term place recognition in natural environments due to gradual changes over time.
        <!-- 
            ###################
            SECTION 8 - Download
            ###################
          -->
        <h2 id="8-Download"><a class="anchor" name="8-Download" href="#8-Download"><span class="octicon octicon-link"></span></a>8. Download</h2>
        <h4 id="81-Checkpoints"><a class="anchor" name="81-Checkpoints" href="#81-Checkpoints"><span class="octicon octicon-link">
        </span></a>8.1 Checkpoint Download</h4>
        <div align="center">
            <table class="tg" border="0">
               <thead>
                  <tr>
                     <th class="tg-0lax">Model</th>
                     <th class="tg-0lax">Checkpoint</th>
                  </tr>
                  </thead>
               <tbody>
                  <tr>
                     <td class="tg-0lax">TransLoc3D<br></td>
                     <td class="tg-0lax"><a href="https://cloudstor.aarnet.edu.au/plus/s/ODFBQ0t7ME1QoJJ" target="_blank" rel="noopener noreferrer">Link</a></td>
                  </tr>
                  <tr>
                     <td class="tg-0lax">MinkLoc3Dv2</td>
                     <td class="tg-0lax"><a href="https://cloudstor.aarnet.edu.au/plus/s/Gi68q66sHlKgt7A" target="_blank" rel="noopener noreferrer">Link</a></td>
                  </tr>
                  <tr>
                     <td class="tg-0lax">LoGG3D-Net</td>
                     <td class="tg-0lax"><a href="https://cloudstor.aarnet.edu.au/plus/s/MCMvtyOGgA87FKD" target="_blank" rel="noopener noreferrer">Link</a></td>
                  </tr>
               </tbody>
            </table>
         </div>
         The links in the above table will allow you to download checkpoints for our trained models on the TransLoc3D, MinkLoc3Dv2 and LoGG3D-Net architectures trained in the paper associated with this dataset release.
         <h4 id="82-Dataset"><a class="anchor" name="82-Dataset" href="#82-dataset"><span class="octicon octicon-link">
         </span></a>8.2 Dataset Download</h4>
         Our dataset can be downloaded through <a href="https://data.csiro.au/collection/csiro:56372?q=wild-places&_st=keyword&_str=1&_si=1"> The CSIRO Data Access Portal</a>.  Detailed instructions for dataset, downloading the dataset can be found in the <a href="https://github.com/csiro-robotics/Wild-Places/blob/Website/images/Wild-Places-SupplementaryMaterial.pdf">Supplementary Material file</a> provided on the in this page and CSIRO's data access portal page.
         <h4 id="83-License"><a class="anchor" name="83-Dataset" href="#83-dataset"><span class="octicon octicon-link">
         </span></a>8.3 License</h4>
         <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">
         <img src="images/license.png"></a>
         <br>We release this dataset under a <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/"> Creative Commons Attribution Noncommercial-Share Alike 4.0 Licence </a>
        <!-- 
            ###################
            END OF OUR CONTENT
            ###################
          -->
