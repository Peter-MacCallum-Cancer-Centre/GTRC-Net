import os
from os.path import join,isdir
import SimpleITK as sitk
import shutil
import gtrc_utils
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

import nnunet_config_paths

start_time=time.time()

"""
Script to run inference based on trained models. For DEEP-PSMA data, only 'psma_pet' and 'fdg_pet' options
will be available once all training scripts have completed (00-06 to pre-process and train nnU-Net and
consensus classifier models for each tracer).

"""


nn_predict_exe='nnUNetv2_predict' #if not available in %PATH update to appropriate location
##nn_predict_exe=r"D:\path\to\python\Scripts\nnUNetv2_predict.exe"

def expand_contract_label(label,distance=5.0):
    """expand or contract sitk label image by distance indicated.
    negative values will contract, positive values expand.
    returns sitk image of adjusted label"""
    lar=sitk.GetArrayFromImage(label)
    label_single=sitk.GetImageFromArray((lar>0).astype('int16'))
    label_single.CopyInformation(label)
    distance_filter = sitk.SignedMaurerDistanceMapImageFilter()
    distance_filter.SetUseImageSpacing(True)
    dmap=distance_filter.Execute(label_single)
    dmap_ar=sitk.GetArrayFromImage(dmap)
    new_label_ar=(dmap_ar<=distance).astype('int16')
    new_label=sitk.GetImageFromArray(new_label_ar)
    new_label.CopyInformation(label)
    return new_label

def run_gtrc_infer(pt,ct,tracer='psma_pet',suv_threshold=3.0,output_fname='gtrc_inferred.nii.gz',
                   return_ttb_sitk=False,temp_dir='temp',fold='all',
                   quick_gtrc=False,quick_expansion_radius=7.):

    if isinstance(pt,str):
        pt_suv=sitk.ReadImage(pt)
    else:
        pt_suv=pt
    if isinstance(ct,str):
        ct=sitk.ReadImage(ct)

##    tracer='psma_pet'  #psma_pet, fdg_pet, or lupsma_spect

    module_dir=os.path.dirname(__file__)

    rs=sitk.ResampleImageFilter()
    rs.SetReferenceImage(pt_suv)
    rs.SetDefaultPixelValue(-1000)
    ct_rs=rs.Execute(ct)
    pt_rs=pt_suv/suv_threshold
    os.makedirs(temp_dir,exist_ok=True) #create/empty nnU-Net temporary dir
    for f in os.listdir(temp_dir):
        if isdir(join(temp_dir,f)):
            shutil.rmtree(join(temp_dir,f))
        else:
            os.unlink(join(temp_dir,f))
    #filenames to end in _0000.nii.gz and _0001.nii.gz, 0=rescaled PET, 1=CT
    os.makedirs(join(temp_dir,'nn_input'),exist_ok=True)
    sitk.WriteImage(pt_rs,join(temp_dir,'nn_input','gtrc_0000.nii.gz'))
    sitk.WriteImage(ct_rs,join(temp_dir,'nn_input','gtrc_0001.nii.gz'))

    call=nn_predict_exe+' -i '+join(temp_dir,'nn_input')+' -o '+join(temp_dir,'nn_output')
    if tracer=='psma_pet':
        call+=' -d '+'801'
    elif tracer=='fdg_pet':
        call+=' -d '+'802'
    elif tracer=='lupsma_spect':
        call+=' -d '+'803'
    call+=' -c 3d_fullres'
    if not fold=='all':
        call+=' -f '+str(fold)
    ##call+=' --save_probabilities'
    ##call+=' -f 0'
    print('images loaded',round(time.time()-start_time,1))
    print('Calling nnU-Net')
    print(call)

    import subprocess
    p=subprocess.Popen(call,stdout=subprocess.PIPE ,stderr=subprocess.PIPE, shell=True)
    output, error=p.communicate()
    print(output,error)

    label=sitk.ReadImage(join(temp_dir,'nn_output','gtrc.nii.gz'))
    lar=sitk.GetArrayFromImage(label)
    local_maxima_threshold,sphere_radius=0.16666,8.
    pt_ar=sitk.GetArrayFromImage(pt_rs)
    tar=(pt_ar>=1.0).astype('int8')

    ct_ar=sitk.GetArrayFromImage(ct_rs)
    pt_ar=sitk.GetArrayFromImage(pt_rs)            
    pred_label=sitk.ReadImage(join(temp_dir,'nn_output','gtrc.nii.gz'))
    pred_ttb_ar=(sitk.GetArrayFromImage(pred_label)==1).astype('int8')
    pred_norm_ar=(sitk.GetArrayFromImage(pred_label)==2).astype('int8')
    
    if not quick_gtrc:
        print('Generating subregions',round(time.time()-start_time,1))
        subregions=gtrc_utils.create_subregion_labels(pt_rs,tar,local_maxima_threshold,sphere_radius)
        sitk.WriteImage(subregions,join(temp_dir,'subregions.nii.gz'))

        columns=['subregion_number','total_volume', 'suv_max', 'suv_mean', 'ct_hu_mean','pred_ttb_overlap','pred_norm_overlap'] 
        training_columns=['total_volume', 'suv_max', 'suv_mean', 'ct_hu_mean','pred_ttb_overlap','pred_norm_overlap'] #which columns to include for multi-variate model
        dfr=pd.DataFrame(columns=columns)

        spacing=subregions.GetSpacing() #get voxel spacing and volume for analysis
        voxel_volume=np.prod(np.array(spacing))/1000. #in ml        
        subregion_ar=sitk.GetArrayFromImage(subregions)

        total_subregions=int(subregion_ar.max()) ##get max value from subregion array for iterating (note some may be empty already)
        print('scoring subregions',spacing,subregions.GetSize(),total_subregions)
        print(round(time.time()-start_time,1))
        for i in range(total_subregions): #iterate through the subregions
            if i%100==0:
                print(i, end=' ')
            subregion_bool=subregion_ar==(i+1)
            total_voxels=(subregion_bool).sum() #count number of voxels in region
            total_volume=total_voxels*voxel_volume #convert to volume (ml/cc)
            if total_volume>0.: #if any volume found compute basic stats
                suv_max=pt_ar[subregion_bool].max() #qspect SUV max
                suv_mean=pt_ar[subregion_bool].mean() #qspect SUV mean
                ct_hu_mean=ct_ar[subregion_bool].mean() #ct Hounsfield Unit mean
        ##                    true_ttb_overlap=(np.logical_and((subregion_bool),(ttb_ar>0.5)).sum())/total_voxels #compute overlap fractions from TTB/Normal predictions for each region
        ##                    true_norm_overlap=(np.logical_and((subregion_bool),(norm_ar>0.5)).sum())/total_voxels #both ground truth and UNet predicted are scored
                pred_ttb_overlap=(np.logical_and((subregion_bool),(pred_ttb_ar>0.5)).sum())/total_voxels
                pred_norm_overlap=(np.logical_and((subregion_bool),(pred_norm_ar>0.5)).sum())/total_voxels
                row=[i+1,total_volume,suv_max,suv_mean,ct_hu_mean,pred_ttb_overlap,pred_norm_overlap]
                dfr.loc[len(dfr)]=row
        dfr.to_csv(join(temp_dir,'subregion_statistics.csv'))


        models=[]
        ##for i in range(1):
        #consensus models located in data/PSMA/subregion_metrics/fold_0/consensus_model.keras 
        if fold=='all':
            for i in range(5):
                if tracer=='psma_pet':
                    model_path=join(module_dir,'data','PSMA','subregion_metrics','fold_'+str(i),'consensus_model.keras')
                elif tracer=='fdg_pet':
                    model_path=join(module_dir,'data','FDG','subregion_metrics','fold_'+str(i),'consensus_model.keras')
                elif tracer=='lupsma_spect':
                    model_path=join(module_dir,'data','LuPSMA','subregion_metrics','fold_'+str(i),'consensus_model.keras')
                model=load_model(model_path,compile=False)
                models.append(model)
        else:
            if tracer=='psma_pet':
                model_path=join(module_dir,'data','PSMA','subregion_metrics','fold_'+str(fold),'consensus_model.keras')
            elif tracer=='fdg_pet':
                model_path=join(module_dir,'data','FDG','subregion_metrics','fold_'+str(fold),'consensus_model.keras')
            elif tracer=='lupsma_spect':
                model_path=join(module_dir,'data','LuPSMA','subregion_metrics','fold_'+str(fold),'consensus_model.keras')
            model=load_model(model_path,compile=False)
            models.append(model)        

        output_ar=np.zeros(subregion_ar.shape)
        included_regions=0
        print('Running Classifier',round(time.time()-start_time,1))
        for i,row in dfr.iterrows():
            label_value=int(row.subregion_number)
            x=row[training_columns].values #create consensus model input variable based on training arrays
            x=np.expand_dims(x,0)
            votes=0
            for model in models:
                pred=model(x).numpy()[0][0] #get prediction [0.0-1.0]
                if pred>0.5:
                    votes+=1
            if votes>(float(len(models))/2):
                output_ar[subregion_ar==label_value]=1
                included_regions+=1
        print('subregions included:',included_regions,'/',len(dfr))

    else:
        pred_ttb_label=sitk.GetImageFromArray(pred_ttb_ar)
        pred_ttb_label.CopyInformation(pred_label)
        pred_ttb_label_expanded=expand_contract_label(pred_ttb_label,distance=quick_expansion_radius)
        pred_ttb_ar_expanded=sitk.GetArrayFromImage(pred_ttb_label_expanded)
        pred_ttb_ar_expanded=np.logical_and(pred_ttb_ar_expanded>0,tar>0)
        
        output_ar=np.logical_and(pred_ttb_ar_expanded>0,pred_norm_ar==0).astype('int8')

    gtrc_voxels=output_ar.sum()
    output_ar[np.logical_and(pred_ttb_ar>0,tar>0)]=1
    added_voxels=output_ar.sum()-gtrc_voxels
    print('nnU-Net Prediction Voxels re-painted:',added_voxels)

    output_label=sitk.GetImageFromArray(output_ar)
    output_label.CopyInformation(pred_label) #pred_ttb_label
    sitk.WriteImage(output_label,join(temp_dir,'output_label.nii.gz'))
    sitk.WriteImage(output_label,output_fname)
    print('Processing complete',round(time.time()-start_time,1))

    if return_ttb_sitk:
        return output_label
    else:
        return
    
##Example Usage:
##ct_fname=r"W:\DEEP-PSMA\CHALLENGE_DATA\train_0001\PSMA\CT.nii.gz"
##pt_fname=r"W:\DEEP-PSMA\CHALLENGE_DATA\train_0001\PSMA\PET.nii.gz"
##suv_threshold=3.0
##
##ttb=run_gtrc_infer(pt_fname,ct_fname,tracer='psma_pet',output_fname=r"W:\DEEP-PSMA\CHALLENGE_DATA\train_0001\PSMA\TTB_gtrc.nii.gz",
##                   return_ttb_sitk=True,
##                   fold=0,suv_threshold=suv_threshold)
