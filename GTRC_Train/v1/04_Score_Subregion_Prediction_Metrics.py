import pandas as pd
import tensorflow.keras as keras
import numpy as np
import os
from os.path import join
import SimpleITK as sitk
from scipy import ndimage
import tensorflow as tf

from tensorflow.keras.models import load_model
from gtrc.file_loader import load_cropped_inference
##from tensorflow_addons.layers import InstanceNormalization

import argparse
parser=argparse.ArgumentParser(description='derives training information for consensus optimisation')
parser.add_argument('--fold_number',type=int,default=1,help='which fold to use #s 1-5',required=False)
args=parser.parse_args()
fold_number=args.fold_number #read in fold number from command line (which subdirectory to evaluate in training dir)

save_inferred_labels=True #whether to save UNet predicted ttb and normal labels. Small files but not necessary for further analysis really
include_background_channel=True
if include_background_channel:
    label_channel=1
else:
    label_channel=0

"""
python 04_Score_Subregion_Prediction_Metrics.py 1
[run for each fold]
"""

data_dir=join('data') #top of data directory
ct_dir=join(data_dir,'ct_lowres') #CT output directory matched to QSPECT resolution
qs_dir=join(data_dir,'qspect_rescaled') #qspect output after scaling intensity to value of 1.0 at min of detected TTB per case
ttb_dir=join(data_dir,'ttb_label_adjusted') #input ground truth tumour burden nifti label
norm_dir=join(data_dir,'norm_label') #output derived normal/physiological nifti - above min threshold but not included in TTB
subregion_dir=join(data_dir,'subregions') #to save subregions for each image based on local max and gradient boundary
extension='.nii.gz' #file extension of input data
data_csv=join(data_dir,'gtrc_train_data.csv') #summary csv of processed data
df_data=pd.read_csv(data_csv,index_col=0)
train_dir='training' #folder to output training/preprocessing information and models
training_csv=join(train_dir,'training_resolution.csv')
fold_dir=join(train_dir,'fold_'+str(fold_number).zfill(2))
region='tumor' #workaround to use training file-loader


if save_inferred_labels: #create output folders if saving inferred labels
    inferred_ttb_dir=join(fold_dir,'ttb_network_inferred')
    if not os.path.exists(inferred_ttb_dir):
        os.mkdir(inferred_ttb_dir)
    inferred_norm_dir=join(fold_dir,'norm_network_inferred')
    if not os.path.exists(inferred_norm_dir):
        os.mkdir(inferred_norm_dir)

for f in os.listdir(fold_dir): #find model files for tumor and normal regions
    if f.startswith('GTRCNet') and f.endswith('.hdf5') and 'normal' in f: #os.path.isdir(join(fold_dir,f))
        norm_model_path=join(fold_dir,f)
        norm_model=load_model(norm_model_path,compile=False)
        if 'QSCT' in f:
            include_ct=True
            input_depth=2
        else:
            include_ct=False
            input_depth=1
    elif f.startswith('GTRCNet') and f.endswith('.hdf5') and 'tumor' in f:
        ttb_model_path=join(fold_dir,f)
        ttb_model=load_model(ttb_model_path,compile=False)

df_training=pd.read_csv(training_csv) #read in resolution details for running inference
x_extent=round(df_training.iloc[0].x_extent,2)
y_extent=round(df_training.iloc[0].y_extent,2)
z_extent=round(df_training.iloc[0].z_extent,2)
xdim=int(df_training.iloc[0].x_resolution)
ydim=int(df_training.iloc[0].y_resolution)
zdim=int(df_training.iloc[0].z_resolution)
resample_dimensions=(zdim,ydim,xdim)
resample_extent=(x_extent,y_extent,z_extent)
input_size=(zdim,ydim,xdim,input_depth)

def load_inference_case(fname): #file loader. Creates TF input array from filename. SITK image at UNet processing resolution also returned to keep spatial information
    qs_path=join(qs_dir,fname)
    if include_ct:
        ct_path=join(ct_dir,fname)
    else:
        ct_path=False
    ttb_path=join(ttb_dir,fname)
    norm_path=join(norm_dir,fname)
    if region=='tumor':
        norm_path=False
    if region=='normal':
        ttb_path=False
    x,im=load_cropped_inference(resample_extent, resample_dimensions, qs_path, ct_path,batch_shape=True)
    return x.astype('float32'),im

df_score=pd.DataFrame(columns=['fold','case','region_num','total_volume', #create dataframe for scoring subregion agreement metrics
                               'suv_max','suv_mean','ct_hu_mean','true_ttb_overlap',
                               'pred_ttb_overlap','true_norm_overlap','pred_norm_overlap'])
counter=0
for j in range(len(df_data)): #loop through each case
    case=df_data.iloc[j].case
    fold=df_data.iloc[j].fold
    print(j,case,fold)
    qs=sitk.ReadImage(join(qs_dir,case)) #read input images and labels
    ttb=sitk.ReadImage(join(ttb_dir,case))
    norm=sitk.ReadImage(join(norm_dir,case))

    rs=sitk.ResampleImageFilter() #create resampler to get intermediate files back to original resolution
    rs.SetReferenceImage(ttb) #use orininal TTB label as spatial reference 
    rs.SetInterpolator(sitk.sitkLinear) #linear initial upsampling of predicted TTB/Norm labels, then clipped at >0.5 for smoother binary output at original resolution
    qs_ar=sitk.GetArrayFromImage(qs) #create numpy arrays for scoring
    ttb_ar=sitk.GetArrayFromImage(ttb)
    norm_ar=sitk.GetArrayFromImage(norm)
    if include_ct: #if ct included load image and create numpy array
        ct=sitk.ReadImage(join(ct_dir,case))
        ct_ar=sitk.GetArrayFromImage(ct)
    
    x,resampled=load_inference_case(case) #load current case in loop
    pred_ttb_resampled_ar=ttb_model(np.expand_dims(x,0)).numpy()[0,...,label_channel] #infer ttb label. Convert to numpy and take only label channel
    pred_ttb_resampled_im=sitk.GetImageFromArray(pred_ttb_resampled_ar) #create SITK image and associate predicted array with U-Net spatial information
    pred_ttb_resampled_im.CopyInformation(resampled)   
    pred_ttb_original_im=rs.Execute(pred_ttb_resampled_im) #resample to original input resolution (linear, continuous [0.0-1.0] prediction)
    pred_ttb_ar=(sitk.GetArrayFromImage(pred_ttb_original_im)>0.5).astype('int8') #clip values >0.5 to binary array
    pred_ttb_original_im=sitk.GetImageFromArray(pred_ttb_ar) #create SITK image and copy spatial information from TTB label
    pred_ttb_original_im.CopyInformation(ttb)

    pred_norm_resampled_ar=norm_model(np.expand_dims(x,0)).numpy()[0,...,label_channel] #same as above but for normal UNet prediction
    pred_norm_resampled_im=sitk.GetImageFromArray(pred_norm_resampled_ar)
    pred_norm_resampled_im.CopyInformation(resampled)
    pred_norm_original_im=rs.Execute(pred_norm_resampled_im)
    pred_norm_ar=(sitk.GetArrayFromImage(pred_norm_original_im)>0.5).astype('int8')
    pred_norm_original_im=sitk.GetImageFromArray(pred_norm_ar)
    pred_norm_original_im.CopyInformation(ttb)
    if save_inferred_labels: #Save upsampled UNET labels if designated
        sitk.WriteImage(sitk.Cast(pred_ttb_original_im,sitk.sitkInt8),join(inferred_ttb_dir,case))
        sitk.WriteImage(sitk.Cast(pred_norm_original_im,sitk.sitkInt8),join(inferred_norm_dir,case))

    qs_spacing=qs.GetSpacing() #get voxel spacing and volume for analysis
    voxel_volume=np.prod(np.array(qs_spacing))/1000. #in ml
    ws_im=sitk.ReadImage(join(subregion_dir,case)) #reads pre-processed subregion image based on watershed/local max rules
    labels_ws=sitk.GetArrayFromImage(ws_im) #convert subregion image to numpy array
    total_subregions=int(labels_ws.max()) ##get max value from subregion array for iterating (note some may be empty already)
    for i in range(total_subregions): #iterate through the subregions
        total_voxels=(labels_ws==(i+1)).sum() #count number of voxels in region
        total_volume=total_voxels*voxel_volume #convert to volume (ml/cc)
        if total_volume>0.: #if any volume found compute basic stats
            suv_max=qs_ar[labels_ws==(i+1)].max() #qspect SUV max
            suv_mean=qs_ar[labels_ws==(i+1)].mean() #qspect SUV mean
            if include_ct:
                ct_hu_mean=ct_ar[labels_ws==(i+1)].mean() #ct Hounsfield Unit mean
            else:
                ct_hu_mean=0. #set to 0 if CT omitted
            
            true_ttb_overlap=(np.logical_and((labels_ws==(i+1)),(ttb_ar>0.5)).sum())/total_voxels #compute overlap fractions from TTB/Normal predictions for each region
            true_norm_overlap=(np.logical_and((labels_ws==(i+1)),(norm_ar>0.5)).sum())/total_voxels #both ground truth and UNet predicted are scored
            pred_ttb_overlap=(np.logical_and((labels_ws==(i+1)),(pred_ttb_ar>0.5)).sum())/total_voxels
            pred_norm_overlap=(np.logical_and((labels_ws==(i+1)),(pred_norm_ar>0.5)).sum())/total_voxels
            row=[fold,case,i+1,total_volume,suv_max,suv_mean,ct_hu_mean,true_ttb_overlap,pred_ttb_overlap,true_norm_overlap,
                 pred_norm_overlap] #save region to new row in dataframe
            df_score.loc[counter]=row
            counter+=1
print(len(df_score),'Subregions analysed. Run Consensus training to complete model setup...')
df_score.to_csv(join(fold_dir,'subregion_prediction_metrics.csv'))  #save all subregion metrics for optimization stage
