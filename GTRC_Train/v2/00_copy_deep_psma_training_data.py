import os
from os.path import join
import shutil

raw_top='CHALLENGE_DATA'
data_top='data'
for case in os.listdir(raw_top):
    print(case)
    for tracer in ['PSMA','FDG']:
        shutil.copyfile(join(raw_top,case,tracer,'CT.nii.gz'),
                        join(data_top,tracer,'CT',case+'.nii.gz'))
        shutil.copyfile(join(raw_top,case,tracer,'PET.nii.gz'),
                        join(data_top,tracer,'PET',case+'.nii.gz'))        
        shutil.copyfile(join(raw_top,case,tracer,'TTB.nii.gz'),
                        join(data_top,tracer,'TTB',case+'.nii.gz'))
        shutil.copyfile(join(raw_top,case,tracer,'threshold.txt'),
                        join(data_top,tracer,'thresholds',case+'.txt')) 
