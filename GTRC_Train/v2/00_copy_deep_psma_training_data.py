import os
from os.path import join
import shutil

raw_top='CHALLENGE_DATA'
data_top='data'
###############################################
# LM: 23-04-2025
###############################################
# List of directories to unpack to
dirs_to_create = [
 './data',
 
 
 './data/PSMA',
 './data/PSMA/CT',
 './data/PSMA/PET',
 './data/PSMA/TTB',
 './data/PSMA/thresholds',
 
 './data/FDG',
 './data/FDG/CT',
 './data/FDG/PET',
 './data/FDG/TTB',
 './data/FDG/thresholds',
    
]

for dir_path in dirs_to_create:
 os.makedirs(dir_path, exist_ok=True)
###############################################
for case in os.listdir(raw_top):
    print(case)
    for tracer in ['PSMA','FDG']:
        shutil.copyfile(join(raw_top,case,tracer,'CT.nii.gz'),
                        join(data_top,tracer,'CT',case+'.nii.gz'))
        shutil.copyfile(join(raw_top,case,tracer,'PET.nii.gz'),
                        join(data_top,tracer,'PET',case+'.nii.gz'))        
        shutil.copyfile(join(raw_top,case,tracer,'TTB.nii.gz'),
                        join(data_top,tracer,'TTB',case+'.nii.gz'))
        shutil.copyfile(join(raw_top,case,tracer,'threshold.json'),
                        join(data_top,tracer,'thresholds',case+'.json')) 
