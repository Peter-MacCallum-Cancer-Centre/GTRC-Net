import os
from os.path import join,isdir
import SimpleITK as sitk
import shutil
import pandas as pd
import numpy as np
import time
import gtrc_utils
import json

##need to have nnUNetv2 set up and executable from path location. Usually fine to set up virtual environment and simply run pip install nnunetv2

input_dataset_folder='data' #location of input training data
##Need subdirectories in the 'data' folder for each tracer
##subdirectories for each tracer are 'CT','PET', and 'TTB', and 'thresholds'

import nnunet_config_paths #sets and exports paths for nnU-Net configuration
training_output_location=nnunet_config_paths.nn_raw_dir #default is 'nnUNet_data/raw'

dataset_values=nnunet_config_paths.dataset_dictionary #default is {'PSMA':801,'FDG':802}, folders named accordingly and corresponds to dataset ID in nnUNet

#set up folders for nn raw training data
psma_folder=join(training_output_location,'Dataset'+str(dataset_values['PSMA'])+'_PSMA_PET') #create PSMA and FDG Subdirectories in nnunet_raw location (PSMA_801)
fdg_folder=join(training_output_location,'Dataset'+str(dataset_values['FDG'])+'_FDG_PET')
print('setting nnunet dataset folders to:\n',psma_folder,'\n',fdg_folder)
os.makedirs(psma_folder,exist_ok=True)
os.makedirs(fdg_folder,exist_ok=True)
for d in [psma_folder,fdg_folder]: #create subdirectories for training images and labels
    os.makedirs(join(d,'imagesTr'),exist_ok=True)
    os.makedirs(join(d,'labelsTr'),exist_ok=True)

subregion_medium_setting=[0.1666,8]

#loop through challenge data, resample, and save to nnunet dataset format

for tracer in ['PSMA','FDG']: #loop through the tracers for each case
    subregion_dir=join(input_dataset_folder,tracer,'subregions')
    mip_dir=join(input_dataset_folder,tracer,'mip_images')
    normal_dir=join(input_dataset_folder,tracer,'normal')
    pet_rescaled_dir=join(input_dataset_folder,tracer,'PET_rescaled')
    ct_resampled_dir=join(input_dataset_folder,tracer,'CT_resampled')
    for d in [subregion_dir,mip_dir,normal_dir,pet_rescaled_dir,ct_resampled_dir]:
        os.makedirs(d,exist_ok=True)
    for case in os.listdir(join(input_dataset_folder,tracer,'CT')): #loop through all cases in trainin directory
        ct=sitk.ReadImage(join(input_dataset_folder,tracer,'CT',case)) #read CT as sitk image
        pt_suv=sitk.ReadImage(join(input_dataset_folder,tracer,'PET',case)) #read PET (units of SUV by default)
        ttb_label=sitk.ReadImage(join(input_dataset_folder,tracer,'TTB',case)) #read TTB label (natively in PET units)
        with open(join(input_dataset_folder,tracer,'thresholds',case.replace('.nii.gz','.txt')),'r') as f:
            suv_threshold=float(f.readlines()[0])
            print('Detected contouring SUV threshold',suv_threshold)
        pt_rescaled=pt_suv/suv_threshold #rescale PET intensity values so 1.0 corresponds to contouring SUV threhold
        ct_resampled=sitk.Resample(ct,pt_suv,sitk.TranslationTransform(3),sitk.sitkLinear,-1000) #resample CT to PET resolution
        ttb_array=sitk.GetArrayFromImage(ttb_label) #0 for background, 1 for above threshold and annotated as disease
        pt_rescaled_array=sitk.GetArrayFromImage(pt_rescaled)
        ttb_normal_array=np.zeros(ttb_array.shape) #create array for multiclass background/tumour/normal label
        ttb_normal_array[ttb_array>0]=1 #set tumour values to 1
        ttb_normal_array[np.logical_and(pt_rescaled_array>=1.0,ttb_array==0)]=2 #"Normal" tissue label consists of voxels above PET threshold and not included in TTB array
        ttb_normal_label=sitk.GetImageFromArray(ttb_normal_array) #create sitk image for nii.gz output
        ttb_normal_label.CopyInformation(ttb_label)  #set spacing/origin/direction from ttb nifti
        normal_array=np.logical_and(pt_rescaled_array>=1.0,ttb_array==0).astype('int8')
        normal_label=sitk.GetImageFromArray(normal_array)
        normal_label.CopyInformation(ttb_label)
        if tracer=='PSMA':
            output_folder=psma_folder
        elif tracer=='FDG':
            output_folder=fdg_folder
        sitk.WriteImage(pt_rescaled,join(output_folder,'imagesTr',,case.replace('.nii.gz','_0000.nii.gz'))) #write rescaled PET image to training image folder as channel 0
        sitk.WriteImage(ct_resampled,join(output_folder,'imagesTr',,case.replace('.nii.gz','_0001.nii.gz'))) #write CT matched to PET resolution image to training image folder as channel 1
        sitk.WriteImage(ttb_normal_label,join(output_folder,'labelsTr',case)) #write output label (background/tumour/normal 0/1/2) to training label folder

        sitk.WriteImage(normal_label,join(normal_dir,case))
        sitk.WriteImage(pt_rescaled,join(pet_rescaled_dir,case))
        sitk.WriteImage(ct_resampled,join(ct_resampled_dir,case))

        #plot MIP image of ttb & normal regions for review
        mip_fname=join(mip_dir,case.replace('.nii.gz','.jpg'))
        gtrc_utils.plot_mip(pt_rescaled,ttb_label,normal_label,mip_fname,title=case,clim_max=2.5,show=False)
        
        #create above-threshold watershed subregions per case
        threshold_array=(pt_rescaled_array>=1.0).astype('int8')
        subregion_im=gtrc_utils.create_subregion_labels(pt_rescaled,threshold_array,subregion_medium_setting[0],subregion_medium_setting[1])
        sitk.WriteImage(subregion_im,join(subregion_dir,case))
        
#create dataset.json file
for d in [psma_folder,fdg_folder]:
    n_images=len(os.listdir(join(d,'labelsTr')))
    json_dict={'channel_names': {'0': 'noNorm', '1': 'CT'}, 'labels': {'background': 0, 'ttb': 1, 'norm': 2}, 'numTraining': n_images, 'file_ending': '.nii.gz'}
    dataset_json_fname=join(d,'dataset.json')
    with open(dataset_json_fname,'w') as f:
        print('writing dataset json file:\n',dataset_json_fname)
        f.write(json.dumps(json_dict, indent=4))

##run nnUNet dataset plan and preprocess
os.system('nnUNetv2_plan_and_preprocess -d 801 -c 3d_fullres --verify_dataset_integrity')
os.system('nnUNetv2_plan_and_preprocess -d 802 -c 3d_fullres --verify_dataset_integrity')

