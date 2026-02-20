from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import colors
from astropy.visualization import make_lupton_rgb
import numpy as np
import pandas as pd
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

import cv2
from detectron2.structures import BoxMode
from astropy.table import Table
import glob
from astropy.coordinates import SkyCoord  # High-level coordinates
from detectron2.config import LazyConfig, get_cfg, instantiate
import os
import scipy.stats as stats
import h5py
import json
import astropy.units as u
from astropy.coordinates import SkyCoord

import warnings
import time

from astropy.wcs import FITSFixedWarning
warnings.filterwarnings("ignore", category=FITSFixedWarning)
import torch
import torch.nn.functional as F
from detectron2.data import detection_utils as utils
import pickle
import detectron2.data as d2data

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cfgfile', type=str,
                    help='path to config file')
parser.add_argument('--run-name', type=str,
                    help='run name')
parser.add_argument('--output-dir', type=str,
                    help='directory that has saved model')

args = parser.parse_args()


with h5py.File('/home/g4merz/DP1/dp1_matched_v4_test.hdf5','r') as f:
    redshifts = f['redshift'][:]
    ras = f['coord_ra'][:]
    decs = f['coord_dec'][:]
    objectIds = f['objectId'][:]
    
testcat = pd.DataFrame({'redshift':redshifts,'ra':ras,'dec':decs,'objectId':objectIds})
testcoords = SkyCoord(ra=ras,dec=decs,unit=u.deg)

cfgfile = args.cfgfile
run_name = args.run_name
output_dir = args.output_dir

with open('/home/g4merz/DP1/processed_data/test_dicts_new.json') as f:
    test_ddicts = json.load(f)

from deepdisc.inference.predictors import AstroPredictor

cfg = LazyConfig.load(cfgfile)
cfg.train.init_checkpoint = os.path.join(output_dir, run_name + ".pth")
cfg.TEST.DETECTIONS_PER_IMAGE = 3000
cfg.model.proposal_generator.anchor_generator.sizes = [[8], [16], [32], [64], [128]]
cfg.model.roi_heads.batch_size_per_image=1024
cfg.model.proposal_generator.post_nms_topk=[6000,3000]
cfg.model.proposal_generator.batch_size_per_image = 1024
for box_predictor in cfg.model.roi_heads.box_predictors:
    box_predictor.test_topk_per_image = 3000
    box_predictor.test_score_thresh = 0.5
    box_predictor.test_nms_thresh = 0.3
   
gf=False
cfg.model.roi_heads.output_features = gf


def get_outputs_withwcs(predictor, metadata, ims, i):
    d = json.loads(metadata[i])
    wcs = d['wcs']
    image = ims[i].reshape(9,d['height'],d['width'])
    with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
        # Apply pre-processing to image.
        # image = self.aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32"))
        inputs = {'image':image, 'wcs':wcs, 'height':image.shape[1], 'width':image.shape[2]}
        out = predictor.model([inputs])
    return out

nondect_inds = []

def match_center(d, outputs):
        
    xs = np.linspace(0, 14, 1401)
    ys = np.zeros_like(xs)
    
    #d = json.loads(test_metadata[ind])
    wcsi = WCS(d['wcs'])
    
    if len(outputs['instances'])==0:
        return None

    gmm = outputs['instances'].pred_gmm.cpu()
    ws = gmm[..., :5]
    ws = F.softmax(ws, dim=-1).numpy()
    mus = gmm[..., 5:10]
    stds = torch.exp(gmm[..., 10:])

    #scores = outputs['instances'].scores.cpu().numpy()

    centers = outputs['instances'].pred_boxes.get_centers().cpu().numpy()
    xs = [center[0] for center in centers]
    ys = [center[1] for center in centers]
    tc = wcsi.pixel_to_world(xs,ys)

        
    idx, d2d, d3d = tc.match_to_catalog_sky(testcoords)    
    if d['img_index'] not in idx:
        print('No matched detection for image ', d['img_index'])
        nondect_inds.append(d['img_index'])
        return None
    
    matches = np.where(idx==d['img_index'])[0]
    dm = d2d[matches]
    dist = dm[dm.value.argmin()].to(u.arcsecond).value
    if dist>1:
        print('No matched detection <1 arcsec for image ', d['img_index'])
        return None

        
    
    mi = matches[dm.value.argmin()]

    xs = np.linspace(0, 11, 1101)
    ys = np.zeros_like(xs)

    pdf = np.zeros_like(xs)
    for i in range(5):
        pdf += stats.norm.pdf(xs, loc=mus[mi].squeeze()[i], scale=stds[mi].squeeze()[i]) * ws[mi].squeeze()[i]
    
    zpred = xs[pdf.argmax()]
    ztrue = redshifts[d['img_index']]
    oid = objectIds[d['img_index']]
    #score = scores[mi]

    gmm = np.array([ws[mi],mus[mi].numpy(),stds[mi].numpy()]) 

    return zpred, ztrue, oid, gmm     


def map_inds(i):
    d = test_ddicts[i]
    wcs = d['wcs']
    
    imgind = int(d['filename'].split('/')[-1].split('.npy')[0][4:])
    filename = f"test_v4_cutouts_{imgind}.npy"
    dirpath = "/home/g4merz/DP1/processed_data/"
    fn = os.path.join(dirpath, filename)
    image = np.load(fn)
    
    instances = utils.annotations_to_instances(d['annotations'], image.shape[1:])
    instances.pred_boxes = instances.gt_boxes
    instances.pred_classes = instances.gt_classes
    instances.gt_id = torch.tensor([a["objectId"] for a in d['annotations']])
    instances.gt_redshift = torch.tensor([a["redshift"] for a in d['annotations']])

    with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
        # Apply pre-processing to image.
        # image = self.aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32"))
        inputs = {'image':image, 'wcs':wcs, 'height':image.shape[1], 'width':image.shape[2], 'annotations':d['annotations'], 'img_index':imgind}
        #out = predictor.model([inputs])
    return inputs, instances

def get_res(dloader,predictor):
    zps = []
    zts = []
    ids = []
    gmms = []
    with torch.no_grad():
        for i, data in enumerate(dloader):
            print(i)
            dataset_dicts = [data[0][0]]
            training_boxes = [data[0][1]]
            batched_outputs = predictor.model.inference(dataset_dicts,training_boxes)
            for i,(d,outputs) in enumerate(zip(dataset_dicts,batched_outputs)):
                outs = match_center(d,outputs)
                if outs is not None: 
                    zp, zt, idi, gmmi = outs
                    zps.append(zp)
                    zts.append(zt)
                    ids.append(idi)
                    #scores.append(si)
                    gmms.append(gmmi)
    zps = np.hstack(zps)
    zts = np.hstack(zts)
    ids = np.hstack(ids)
    #scores = np.hstack(scores)
    gmms = np.vstack(gmms)
    
    return zps,zts,ids, gmms


predictor = AstroPredictor(cfg)


loader = d2data.build_detection_test_loader(
    np.arange(len(objectIds)), mapper=map_inds, batch_size=1
    #np.arange(20), mapper=map_inds, batch_size=1

) 


t0 = time.time()
outs = get_res(loader,predictor)

print('Took ', time.time()-t0, ' seconds')

zps,zts,ids,gmms = outs

#np.save(f'/home/g4merz/DP1/estimation/{run_name}_nondect_inds.npy',np.array(nondect_inds))

test_dict={'z_pred':zps, 'z_spec':zts, 'ids':ids, 'gmms':gmms}

with open(f'/home/g4merz/DP1/estimation/{run_name}_test_center_fixed_outs.npy', 'wb') as fp:
    pickle.dump(test_dict, fp)



