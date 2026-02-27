# DeepDISC_DP1
Collection of code for running DeepDISC on DP1 data


## Running on large cutouts:

`DP1_Access.ipynb` - grabs images, catalogs on NERSC  

`prep_dp1_for_dd.ipynb` - cuts up images into smaller cutouts, saves metadata (filenames, WCS, etc) into DeepDISC-readable format  

`prep_dp1_for_dd_with_scarlet.ipynb` - same as `prep_dp1_for_dd.ipynb` but includes annotations with LSST pipelines scarlet-lite

`dp1_inference.ipynb` - loads a trained model and images and produces DeepDISC detections  


## Running on small cutouts: 

`DP1_photo-z_cutouts_custom_scarlet.ipynb` - runs custom DeepDISC scarlet implementation on DP1 cutouts, with a known detection catalog

`DP1_photo-z_cutouts_custom_scarlet.ipynb` - runs LSST scarlet-lite implementation on DP1 cutouts, with LSST detection catalog

`Estimation.ipynb`  - plots loss curves, prediction outputs for an example image, and gathers all saved predictions to plot zspec vs zphot

`DC2_photo-z_model_inference.ipynb `- loads a trained model and images and produces DeepDISC detections  