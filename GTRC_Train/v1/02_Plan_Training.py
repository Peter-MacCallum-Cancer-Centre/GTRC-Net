import os
from os.path import join
import numpy as np
import SimpleITK as sitk
import random
import pandas as pd
import ast

"""
Images are always processed at the same resolution and physical extent
in space. This portion of the preprocessing step will evaluate the training
data and try to determine a reasonable extent that:
1) encompasses the physical space of most if not all of the training data
2) tries to match a common spacing (half-spacing) to the majority of
input data to reduce resampling effects, but this is probably not terribly
important as it only affects the AI inference stage and all post processing
is done at the native image resolution

If any issues with results from running this script, easiest workaround
is to manually enter in sensible resolution and spacing details in the
output csv (training/training_resolution.csv). That file informs the training
and inference.

"""

training_resolution=(128,128,256) #resolution to aim for

data_dir=join('data') #top of data directory
ct_dir=join(data_dir,'ct_lowres') #CT output directory matched to QSPECT resolution
qs_dir=join(data_dir,'qspect_rescaled') #qspect output after scaling intensity to value of 1.0 at min of detected TTB per case
ttb_dir=join(data_dir,'ttb_label_adjusted') #input ground truth tumour burden nifti label
norm_dir=join(data_dir,'norm_label') #output derived normal/physiological nifti - above min threshold but not included in TTB
extension='.nii.gz' #file extension of input data
data_csv=join(data_dir,'gtrc_train_data.csv') #summary csv of processed data


df_data=pd.read_csv(data_csv,index_col=0)

train_dir='training' #folder to output training/preprocessing information and models
training_csv=join(train_dir,'training_resolution.csv')
if not os.path.exists(train_dir):
    os.mkdir(train_dir)


#"613.7855835 613.7855835 1227.57116672" "128 128 256" test run extent and resolution
extents=[]
resolutions=[]
spacings=[]
for i in range(len(df_data)):
    extents.append(ast.literal_eval(df_data.iloc[i]['extent']))
    resolutions.append(ast.literal_eval(df_data.iloc[i]['resolution']))
    spacings.append(ast.literal_eval(df_data.iloc[i]['spacing']))
extents=np.array(extents)
resolutions=np.array(resolutions)
spacings=np.array(spacings)

#get most frequent x,y,z extents
x_extent=np.argmax(np.bincount((extents[:,0]*1e5).astype('int64')))/1E5
y_extent=np.argmax(np.bincount((extents[:,1]*1e5).astype('int64')))/1E5
z_extent=np.argmax(np.bincount((extents[:,2]*1e5).astype('int64')))/1E5

#for z extent, this is often variable in PET and is better chosen to
#accommodate the majority of patients
z_90th_extent=np.percentile(extents[:,2],90) #find 90th percentile of z-extents
z_max_extent=extents[:,2].max()

x_spacing=x_extent/training_resolution[0] #determine x/y spacing to match common axial extent
y_spacing=x_extent/training_resolution[1]

#check if using same spacing for z-axis would include most cases
if x_spacing*training_resolution[2]>z_90th_extent:
    z_spacing=x_spacing #will use isotropic voxel spacing
else:
    z_spacing=z_90th_extent/training_resolution[2] #otherwise use spacing to match 90th percentile of z-extents
    #note: if a small amount of image space is clipped from z-axis in
    #some cases it likely won't affect final segmentation output to
    #an appreciable degree. The PET/SPECT field of view will often be
    #chosen with some additional range and the consensus network may
    #still appropriately decide to include/omit these regions in post-processing
z_training_extent=z_spacing*training_resolution[2]
    
df_training=pd.DataFrame(columns=['x_extent','y_extent','z_extent',
                                  'x_resolution','y_resolution','z_resolution',
                                  'x_spacing','y_spacing','z_spacing',
                                  'subregion_local_max','subregion_radius'])
df_training.loc[0]=[x_extent,y_extent,z_training_extent,
                    training_resolution[0],training_resolution[1],training_resolution[2],
                    x_spacing,y_spacing,z_spacing,
                    df_data.iloc[0]['subregion_local_max'],df_data.iloc[0]['subregion_radius']]
df_training.to_csv(training_csv)
    
