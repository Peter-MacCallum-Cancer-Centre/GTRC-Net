import os
from os.path import join


nnunet_data_folder=join(os.path.dirname(__file__),'data','nnUNet_data') #default nnUnet dataset folder, relative to this file location
print('configuring nnUNet Paths (raw, preprocessed, results) in:',nnunet_data_folder)

nn_raw_dir=join(nnunet_data_folder,'raw') #define and create subdirectories for raw/preprocessed/results
nn_preprocessed_dir=join(nnunet_data_folder,'preprocessed')
nn_results_dir=join(nnunet_data_folder,'results')
os.makedirs(nnunet_data_folder,exist_ok=True)
os.makedirs(nn_raw_dir,exist_ok=True)
os.makedirs(nn_preprocessed_dir,exist_ok=True)
os.makedirs(nn_results_dir,exist_ok=True)


os.environ["nnUNet_raw"] = nn_raw_dir #set variables (equivalent of "export nnUNet_raw=[nn_raw_dir]")
os.environ["nnUNet_preprocessed"] = nn_preprocessed_dir
os.environ["nnUNet_results"] = nn_results_dir


dataset_dictionary={'PSMA':801,'FDG':802}
