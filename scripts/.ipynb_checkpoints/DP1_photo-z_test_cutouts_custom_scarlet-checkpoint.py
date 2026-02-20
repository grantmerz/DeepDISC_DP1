import sys
import deepdisc

# Standard imports
import sys, os
import numpy as np
import time
import glob

import scarlet
import sep

import astropy.io.fits as fits
from astropy.wcs import WCS
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.coordinates import SkyCoord

from scarlet.display import AsinhMapping
from astropy.nddata import Cutout2D

# DeepDISC imports
import deepdisc.preprocessing.detection as detection
import deepdisc.preprocessing.process as process

import matplotlib
import matplotlib.pyplot as plt

# use a better colormap and don't interpolate the pixels
#matplotlib.rc('image', cmap='gray', interpolation='none', origin='lower')
from skimage.util.shape import view_as_blocks
from matplotlib import colors

import pandas as pd
import h5py
import json
from astropy.visualization import make_lupton_rgb
from deepdisc.data_format.file_io import DDLoader
from deepdisc.data_format.annotation_functions.annotate_dp1 import annotate_dp1
from deepdisc.data_format import conversions
from deepdisc.data_format.conversions import fitsim_to_numpy
from deepdisc.data_format.file_io import convert_to_json, get_data_from_json

with h5py.File('/home/g4merz/DP1/dp1_matched_v4_test.hdf5','r') as f:
    redshifts = f['redshift'][:]
    ras = f['coord_ra'][:]
    decs = f['coord_dec'][:]
    objectIds = f['objectId'][:]


outdir = '/home/g4merz/DP1/processed_data/test/'



#This function is specific to the data format. You will need your own function to load custom data

def generate_training_data_example(sp, outdir='/home/g4merz/DP1/processed_data/', plot_image=False, plot_stretch_Q=False, plot_scene=False,
                                   plot_likelihood=False, write_results=True, filters = ['u','g','r','i','z','y']):
    """
    Parameters
    ----------
    c : SkyCoord object
          The ra, dec pointing (single or lists of pointings)
    plot_image : bool
          Whether or not to plot the image
    plot_stretch_Q : bool
          Whether or not to plot different normalizations of your image using the stretch, Q parameters.
    plot_scene : bool
           Whether or not plot scene with scarlet
    plot_likelihood : bool
           Whether or not plot the log likelihood of the scarlet fitting
    write_results : bool
          Whether or not to write results to FITS file
    cutout_size : [int, int]
          Cutout shape of image
          
    Returns
    -------
    The scarlet image test in FITS files.
    
    """

        
    ### Run scarlet on image ###
    datas = np.load(f'/home/g4merz/DP1/processed_data/test_v4_cutouts_{sp}.npy')
    catalog = pd.read_csv(f'/home/g4merz/DP1/catalogs/test_v4_cutout_catalog_{sp}.csv')

    # Image pixel scale in arcsec/pixel
    ps = 0.2
    # Approximate PSF size, you can use a PSF image instead
    sigma_obs = gaussian_fwhm_to_sigma*0.8/ps
       
    # Run Scarlet
    out = detection.run_scarlet(datas, filters, catalog=catalog, lvl=2, sigma_model=1, sigma_obs=sigma_obs, psf=None, plot_scene=plot_scene,
                         max_chi2=1000000, morph_thresh=1, stretch=15, Q=10, 
                         plot_wavelet=False, plot_likelihood=plot_likelihood, plot_sources=False, add_ellipses=True,
                         add_labels=False, add_boxes=False, lvl_segmask=2, maskthresh=0.005,return_models=False, percentiles=(30,90))

    # Unpack output
    observation, starlet_sources, model_frame, catalog, segmentation_masks = out

    
    # Save Scarlet data to FITS file
    if write_results:
        filenames = process.write_scarlet_results_nomodels(datas, observation, starlet_sources, model_frame, 
                                             segmentation_masks, outdir=outdir, 
                                             filters=filters, s=f'{sp}', catalog=catalog, keys=['objectId'])
    
        print(f'\nSaved scarlet results as {filenames} \n')
    
    #return out
        
        
        
t0 = time.time()

inds = np.arange(len(objectIds))
#inds = np.arange(24)


from functools import partial

import multiprocessing as mp
with mp.Pool(processes=64) as pool:
    results = pool.map(partial(generate_training_data_example,outdir=outdir),inds)
    
print(time.time()-t0,' seconds')


#outdir = '/home/g4merz/DP1/processed_data/'

loader = DDLoader().generate_filedict(outdir, ['u', 'g', 'r', 'i', 'z','y'], '*img*.fits', '*mask*',filt_loc=0)
filedict = loader.filedict
img_files = np.transpose([filedict[filt]["img"] for filt in filedict["filters"]])


dataset_dicts=[]
dataset_dicts = loader.generate_dataset_dict(annotate_dp1,dirpath=outdir).get_dataset()  

dfile = os.path.join(outdir,'test_dicts.json')
convert_to_json(dataset_dicts, dfile)

#for file in glob.glob(os.path.join(outdir,'*.fits')):
#    os.remove(file)