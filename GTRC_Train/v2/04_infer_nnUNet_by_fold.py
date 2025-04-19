import json
import os
from os.path import join,isdir
import pandas as pd

###Script to determine appropriate validation cases for training subregion
###consensus MLP
###can only be run AFTER nnUNet has been run for the first time


input_dataset_folder='data' #location of input training data
import nnunet_config_paths #sets and exports paths for nnU-Net configuration
nn_raw_dir=nnunet_config_paths.nn_raw_dir #default is 'nnUNet_data/raw'
nn_result_dir=nnunet_config_paths.nn_results_dir #default is 'nnUNet_data/raw'
dataset_values=nnunet_config_paths.dataset_dictionary
nn_inferred_top=join(input_dataset_folder,'nn_inferred')
for tracer in list(dataset_values):
    os.makedirs(join(nn_inferred_top,tracer),exist_ok=True)
n_folds=1 #set at 1 for testing, flip to 5

for tracer in list(dataset_values): #'PSMA' or 'FDG'
    raw_dataset_folder=join(nn_raw_dir,'Dataset'+str(dataset_values[tracer])+'_'+tracer+'_PET','imagesTr')
    for i in range(n_folds):
        nn_inferred_dir=join(nn_inferred_top,tracer,'fold_'+str(i))
        os.makedirs(nn_inferred_dir,exist_ok=True)
        call='nnUNetv2_predict -i '+raw_dataset_folder                           
        call+=' -o '+nn_inferred_dir
        call+=' -c 3d_fullres -d '+str(dataset_values[tracer])
        call+=' -f '+str(i)
        print(call)
        os.system(call)

