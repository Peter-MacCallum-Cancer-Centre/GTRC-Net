import pandas as pd
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import os
from os.path import join
import SimpleITK as sitk
import sys
import argparse
from tensorflow.keras.callbacks import CSVLogger
import datetime
####
from gtrc.file_loader import load_cropped
from gtrc.model import BuildModel
from gtrc import unified_focal_loss_functions as uni_losses
import shutil

"""

python 03_Train_Segmentation_Nets.py --region tumor --fold_number 1 --starting_depth 8 --n_epochs 1000 --include_ct True --force --continue
python 03_Train_Segmentation_Nets.py --region normal --fold_number 1 --starting_depth 8 --n_epochs 1000 --include_ct True --force --continue

                                        [extent] [resolution] [starting depth] [region] [loss] [include_ct] [use_pyramid_pooling] [fold#]

"""


batch_size=1 #number of image volumes to include per batch
include_background_channel=True #whether to include background channel in final layer (eg one-hot labels) 
if include_background_channel:
    starting_channel=1
else:
    starting_channel=0
def str2bool(v): #bit of code for managing argparse to boolean (t/f)
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser=argparse.ArgumentParser(description='whole body tumour/normal segmentor') #CLI arguments
parser.add_argument("--region",type=str,help="Region to segment either tumour/normal") #Region is required to manage tumor/normal branches, but others can be skipped as needed
parser.add_argument('--fold_number',type=int,default=1,help='which fold to use #s 1-5',required=False)
parser.add_argument('--starting_depth',type=int,default=8,help='Depth of first channel, deeper layers all increased by multiple of 2x eg [16, 32, 64, 128, 256]',required=False)
parser.add_argument('--n_epochs',type=int,default=1000,help='Number of epochs for training',required=False)
parser.add_argument("--include_ct", type=str2bool, nargs='?', default=True, help="Include CT for as well as QSPECT",required=False) #Should probably include as will always be available
parser.add_argument('--force','-f', action="store_true", help="whether to force training if existing modelfilename exists",default=False)
parser.add_argument('--continue_training','-c',action="store_true", help="Whether to continue training from previous epoch if available",default=False)

args=parser.parse_args()

include_ct=args.include_ct
if include_ct: #if ct included append to string to output model file name and change number of input channels
    ct_string='QSCT'
    input_depth=2
else:
    ct_string='QS'
    input_depth=1
starting_depth=int(args.starting_depth)
n_epochs=args.n_epochs
starting_depth=args.starting_depth
fold_number=args.fold_number
force_training=args.force
continue_training=args.continue_training

region=args.region
if 'norm' in region.lower():
    region='normal'
elif 'tumour' in region.lower() or 'tumor' in region.lower() or 'ttb' in region.lower():
    region='tumor'
else:
    print('Region must be called as either normal, tumour/tumor')
    sys.exit()


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
fold_dir=join(train_dir,'fold_'+str(fold_number).zfill(2)) #location of per-fold output data. Will encompass all of the results and necessary for creating portable trained model
if not os.path.exists(fold_dir):
    os.mkdir(fold_dir)
try:
    df_training=pd.read_csv(training_csv) #load pre-processed training data information
except:
    print('Run preprocessing and planning scripts before running training')
    sys.exit()

##loss and scoring metrics
sym_unified_focal_loss=uni_losses.sym_unified_focal_loss() #Taken from https://github.com/mlyg/unified-focal-loss - https://arxiv.org/pdf/2102.04525v4.pdf
def non_intersecting(y_true,y_pred): #metrics and standard dice loss calculation
    return K.sum(tf.abs(tf.subtract(y_true[...,starting_channel:],y_pred[...,starting_channel:])))
def agreement(y_true,y_pred):
    return K.sum(tf.multiply(y_true[...,starting_channel:],y_pred[...,starting_channel:]))
def standard_dice(y_true,y_pred): #dice only calculated on label channel if BG included in y array
    return (2* agreement(y_true,y_pred))/ (K.sum(y_true[...,starting_channel:]) + K.sum(y_pred[...,starting_channel:]))
def dice_loss(y_true,y_pred):
    return -standard_dice(y_true,y_pred)
def accuracy(y_true,y_pred):
    return (agreement(y_true,y_pred)-non_intersecting(y_true,y_pred))/(K.sum(y_true[...,starting_channel:]))
#resampling details
x_extent=round(df_training.iloc[0].x_extent,2) #populated from preprocessing data csv
y_extent=round(df_training.iloc[0].y_extent,2)
z_extent=round(df_training.iloc[0].z_extent,2)
xdim=int(df_training.iloc[0].x_resolution)
ydim=int(df_training.iloc[0].y_resolution)
zdim=int(df_training.iloc[0].z_resolution)
resample_dimensions=(zdim,ydim,xdim)
resample_extent=(x_extent,y_extent,z_extent)
input_size=(zdim,ydim,xdim,input_depth)
print('input_size',input_size)
#output filename details
spacing_string=str(x_extent)+'_'+str(y_extent)+'_'+str(z_extent)+'-'+str(xdim)+'_'+str(ydim)+'_'+str(zdim) #for keeping a few relevant details in the model filenames
filter_depths=(starting_depth*np.array([1,2,4,8,16])).tolist()
fname='GTRCNet'+'-'+spacing_string+'-'+region+'-'+ct_string
print(fname)
model_filename=join(fold_dir,fname+'.hdf5')
if os.path.exists(model_filename) and not force_training:
    print('existing trained model file already exists:',model_filename)
    print('Backup file to another location or re-run with force training flag (--force/-f) continue')
    sys.exit()
progress_model_dir=join(fold_dir,fname+'_in_training')
if not continue_training:
    if os.path.exists(progress_model_dir):
        shutil.rmtree(progress_model_dir)
#quick summary
print('starting feature depth and list:',starting_depth, filter_depths)
print('Target region:',region)
print('Include CT in training?',include_ct)
print('Fold number:',fold_number)
#separate train/test cases
dftrain=df_data[df_data.fold!=fold_number] #separate train/validation sets based on fold number argument
dftest=df_data[df_data.fold==fold_number]
print('N Tranining/Testing cases',len(dftrain),len(dftest))
training_cases=dftrain.case.values
testing_cases=dftest.case.values
def load_training_case(fname): #file loader function for tf-dataset. Dataset is first created from list of case names which are then mapped through file-loader functions
    fname=fname.numpy().decode('utf-8') #convert tensor string to text
    qs_path=join(qs_dir,fname) #few global path variables from above
    if include_ct: #if ct indicated will load, otherwise passing false to load_cropped will skip the input channel
        ct_path=join(ct_dir,fname)
    else:
        ct_path=False
    ttb_path=join(ttb_dir,fname) #tumor segmentation path
    norm_path=join(norm_dir,fname) #generated normal segmentation path
    if region=='tumor': #legacy from testing single network that predicted both labels
        norm_path=False
    if region=='normal':
        ttb_path=False
    #big file-loader function included in secondary module. Loads images, performs data augmentation (as indicated) and resamples to network input dimensions
    x,y=load_cropped(resample_extent, resample_dimensions, qs_path, ttb_path, ct_path,norm_path, augment=True,
                              include_background_channel=include_background_channel,batch_shape=True,crop_augment=True)
    return x.astype('float32'),y.astype('float32') #return x,y arrays 
def load_testing_case(fname): #same function as above, but with no data augmentation
    fname=fname.numpy().decode('utf-8')
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
    x,y=load_cropped(resample_extent, resample_dimensions, qs_path, ttb_path, ct_path,norm_path, augment=False,include_background_channel=include_background_channel,batch_shape=True)
    return x.astype('float32'),y.astype('float32')
#create dataset objects
train_ds = tf.data.Dataset.from_tensor_slices(training_cases) #list of filenames gets converted to tensors
#when case is ready to be loaded, gets mapped through load_training_case(fname) function and returned as x,y tensors
train_ds = train_ds.map(lambda item: tf.py_function(load_training_case, [item], [tf.float32,tf.float32]),num_parallel_calls=tf.data.AUTOTUNE) #NOTE: set num_parallel_calls to 1 if memory error...
train_ds_batch=train_ds.batch(batch_size)
train_ds_batch = train_ds_batch.prefetch(tf.data.AUTOTUNE)

#same but for test/validation cases
test_ds = tf.data.Dataset.from_tensor_slices(testing_cases)
test_ds = test_ds.map(lambda item: tf.py_function(load_testing_case, [item], [tf.float32,tf.float32]),num_parallel_calls=tf.data.AUTOTUNE)
test_ds_batch=test_ds.batch(batch_size)
test_ds_batch = test_ds_batch.prefetch(tf.data.AUTOTUNE)

#some training details


steps_per_epoch=int(np.ceil(len(dftrain)/batch_size))
validation_steps=int(np.ceil(len(dftest)/batch_size))
csv_logger = CSVLogger(join(fold_dir,fname+'.csv'),append=True) #location for training history CSV
checkpointer = ModelCheckpoint(model_filename, save_best_only=True, mode='max', monitor='val_standard_dice') #location for saving best model file
##backup_callback=tf.keras.callbacks.BackupAndRestore(
##    progress_model_dir, save_freq="epoch", delete_checkpoint=True, save_before_preemption=False
##) #may be located in tf.keras.callbacks.BackupAndRestore... (no experimental depending on version)
backup_callback=tf.keras.callbacks.experimental.BackupAndRestore(progress_model_dir)

adam=keras.optimizers.Adam(lr=0.0001,beta_1=0.9,beta_2=0.999,epsilon=1e-8,decay=0.0) #basid adam optimiser
model=BuildModel(input_size=input_size, negative_slope=0., #builds model file, similar to standard 3D U-net
                            filter_depths=filter_depths,output_channels=2,p=0.2)
model.compile(optimizer=adam, loss=sym_unified_focal_loss, metrics=[non_intersecting,agreement,standard_dice,
                                                         accuracy,sym_unified_focal_loss,]) #build model
model.summary() #print layers to screen
callbacks=[checkpointer,csv_logger,backup_callback] #set training callbacks for saving model and logs
#Run training
history=model.fit(x=train_ds_batch,validation_data=test_ds_batch, epochs=n_epochs, callbacks=callbacks,steps_per_epoch=steps_per_epoch,validation_steps=validation_steps)  #(available_cases-validation_size)
f=open(join(fold_dir,region+'_complete.txt'),'w')#once finished creates empty file to show that training has completed
f.close()
