import json
import os
from os.path import join,isdir
import pandas as pd

###Script to determine appropriate validation cases 
###for training subregion consensus MLP
###can only be run AFTER nnUNet has been run for the first time

input_dataset_folder='data' #location of input training data
import nnunet_config_paths #sets and exports paths for nnU-Net configuration
training_output_location=nnunet_config_paths.nn_preprocessed_dir #default is 'nnUNet_data/raw'
dataset_values=nnunet_config_paths.dataset_dictionary #default is {'PSMA':801,'FDG':802}, folders named accordingly and corresponds to dataset ID in nnUNet

#set up folders for nn raw training data
psma_folder=join(training_output_location,'Dataset'+str(dataset_values['PSMA'])+'_PSMA_PET') #create PSMA and FDG Subdirectories in nnunet_raw location (PSMA_801)
fdg_folder=join(training_output_location,'Dataset'+str(dataset_values['FDG'])+'_FDG_PET')

tracers=['PSMA','FDG'] #hard coded for DEEP-PSMA not taken from config_paths file...
nn_folders=[psma_folder,fdg_folder]
for i in range(2):
    splits_final_fname=join(nn_folders[i],"splits_final.json")
    # Open and read the JSON file
    with open(splits_final_fname, 'r') as file:
        splits_final = json.load(file)
    df=pd.DataFrame(columns=['case','val_fold'])
    for j in range(len(splits_final)):
        val_cases=splits_final[j]['val']
        for case in val_cases:
            df.loc[len(df)]=[case,j]
    validation_csv_fname=join(input_dataset_folder,tracers[i],'validation_folds.csv')
    print('writing validation splits to:',validation_csv_fname)
    df.to_csv(validation_csv_fname)
