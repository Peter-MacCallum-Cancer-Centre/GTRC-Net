import pandas as pd
import numpy as np
import os
from os.path import join
import SimpleITK as sitk
from scipy import ndimage
import time, random, timeit, sys


input_dataset_folder='data' #location of input training data
import nnunet_config_paths #sets and exports paths for nnU-Net configuration
dataset_values=nnunet_config_paths.dataset_dictionary
nn_inferred_top=join(input_dataset_folder,'nn_inferred')

n_folds=1 #set at 1 for testing, flip to 5

def shuffle_remaining_cases(case_list_nii,score_dir):
    remaining_cases=[]
    score_dir_cases=[]
    for case in os.listdir(score_dir):
        score_dir_cases.append(case.replace('.csv','.nii.gz'))
    for case in case_list_nii:
        if case not in score_dir_cases:
            remaining_cases.append(case)
    random.shuffle(remaining_cases)
    return remaining_cases

for tracer in list(dataset_values):
    os.makedirs(join(nn_inferred_top,tracer),exist_ok=True)
    data_dir=join(input_dataset_folder,tracer)
    subregion_metric_dir=join(data_dir,'subregion_metrics')
    ct_dir=join(data_dir,'CT_resampled')
    pt_dir=join(data_dir,'PET_rescaled')
    subregion_dir=join(data_dir,'subregions')
    ttb_dir=join(data_dir,'TTB')
    norm_dir=join(data_dir,'normal')
    data_csv=join(data_dir,'validation_folds.csv')
    df_data=pd.read_csv(data_csv,index_col=0)
    cases=df_data.case.values
    ##fold_numbers=['0','1','2','3','4']
    ##fold_numbers=['0']

    for fold_number in range(n_folds):
        fold_number=str(fold_number)
        results_dir=join(subregion_metric_dir,'fold_'+fold_number)
        os.makedirs(results_dir,exist_ok=True)
        score_dir=join(results_dir,'case_scores')
        os.makedirs(score_dir,exist_ok=True)
   
        inferred_dir=join(input_dataset_folder,'nn_inferred',tracer,'fold_'+fold_number)
        while len(os.listdir(score_dir))<len(cases): 
            #loop to calculate subregion overlap statistics per-case until all have been processed
            #shuffling facilitates splitting across multiple compute jobs, useful for larger dataset
            remaining_cases=shuffle_remaining_cases(cases,score_dir)
            case=remaining_cases[0]
            
            df_score_case=pd.DataFrame(columns=['fold','case','region_num','total_volume', #create dataframe for scoring subregion agreement metrics
                                           'suv_max','suv_mean','ct_hu_mean','true_ttb_overlap',
                                           'pred_ttb_overlap','true_norm_overlap','pred_norm_overlap'])              
            if True:
                print(len(os.listdir(score_dir))+1,'/',len(cases),end='\t')
                print('remaining cases:',len(remaining_cases),end='\t')
                start=time.time()
                fold=str(df_data[df_data.case==case].val_fold.values[0])
                print(case,fold)
                subregions=sitk.ReadImage(join(subregion_dir,case))
                spacing=subregions.GetSpacing() #get voxel spacing and volume for analysis
                voxel_volume=np.prod(np.array(spacing))/1000. #in ml        
                subregion_ar=sitk.GetArrayFromImage(subregions)
                ttb_ar=sitk.GetArrayFromImage(sitk.ReadImage(join(ttb_dir,case)))
                norm_ar=sitk.GetArrayFromImage(sitk.ReadImage(join(norm_dir,case)))
                ct_ar=sitk.GetArrayFromImage(sitk.ReadImage(join(ct_dir,case)))
                pt_ar=sitk.GetArrayFromImage(sitk.ReadImage(join(pt_dir,case)))
                pred_label=sitk.ReadImage(join(inferred_dir,case))
                pred_ttb_ar=(sitk.GetArrayFromImage(pred_label)==1).astype('int8')
                pred_norm_ar=(sitk.GetArrayFromImage(pred_label)==2).astype('int8')
                
                total_subregions=int(subregion_ar.max()) ##get max value from subregion array for iterating (note some may be empty already)
                print(spacing,subregions.GetSize(),total_subregions)
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
                        true_ttb_overlap=(np.logical_and((subregion_bool),(ttb_ar>0.5)).sum())/total_voxels #compute overlap fractions from TTB/Normal predictions for each region
                        true_norm_overlap=(np.logical_and((subregion_bool),(norm_ar>0.5)).sum())/total_voxels #both ground truth and UNet predicted are scored
                        pred_ttb_overlap=(np.logical_and((subregion_bool),(pred_ttb_ar>0.5)).sum())/total_voxels
                        pred_norm_overlap=(np.logical_and((subregion_bool),(pred_norm_ar>0.5)).sum())/total_voxels
                        row=[fold,case,i+1,total_volume,suv_max,suv_mean,ct_hu_mean,true_ttb_overlap,pred_ttb_overlap,true_norm_overlap,
                             pred_norm_overlap] #save region to new row in dataframe
                        df_score_case.loc[len(df_score_case)]=row    
                print('case completed in:',round(time.time()-start,1),'seconds')
                df_score_case.to_csv(join(score_dir,case.replace('.nii.gz','.csv')))
        print('all cases complete, compiling metrics to:',join(results_dir,'subregion_prediction_metrics.csv'))
        df_score=pd.DataFrame(columns=['fold','case','region_num','total_volume', #create dataframe for scoring subregion agreement metrics
                                       'suv_max','suv_mean','ct_hu_mean','true_ttb_overlap',
                                       'pred_ttb_overlap','true_norm_overlap','pred_norm_overlap'])
        for case_csv in os.listdir(score_dir):
            df_score_case=pd.read_csv(join(score_dir,case_csv))
            df_score=pd.concat([df_score,df_score_case],ignore_index=True)
        df_score_csv=join(results_dir,'subregion_prediction_metrics.csv')
        df_score.to_csv(df_score_csv)

