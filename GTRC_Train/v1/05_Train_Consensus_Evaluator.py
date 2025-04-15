import pandas as pd
import numpy as np
import os
from os.path import join
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint
import argparse

"""
python 05_Train_Consensus_Evaluator.py --fold_number 1
"""
parser=argparse.ArgumentParser(description='derives training information for consensus optimisation')
parser.add_argument('--fold_number',type=int,default=1,help='which fold to use #s 1-5',required=False)
parser.add_argument('--force','-f', action="store_true", help="whether to force training if previous model filename exists",default=False)

args=parser.parse_args()
fold_number=args.fold_number #get fold number for analysis
train_dir='training' #folder to output training/preprocessing information and models
fold_dir=join(train_dir,'fold_'+str(fold_number).zfill(2))
df_score=pd.read_csv(join(fold_dir,'subregion_prediction_metrics.csv'),index_col=0) #read in overlap and scoring metrics from previous script
n_epochs=500 #number of epochs to run consensus model
create_final_labels=True #whether to save final prediction labels and score dice metrics
render_labels=True
final_label_dir=join(fold_dir,'final_labels')
render_dir=join(fold_dir,'prediction_renders')
data_dir=join('data') #top of data directory
qs_dir=join(data_dir,'qspect_rescaled') #qspect output after scaling intensity to value of 1.0 at min of detected TTB per case
model_path=join(fold_dir,'consensus_model.hdf5')
force_training=args.force
if os.path.exists(model_path) and not force_training:
    print('Existing consensus model found:',model_path)
    print('Back up to new location or re-run with force training flag (--force/-f) train updated model')
    print('proceeding with inference and final statistics...')
    skip_consensus_training=True
else:
    skip_consensus_training=False
if render_labels: #put in conditional for optional config with vtk
    #from gtrc import render_vtk_label_accuracy_function as render_vtk
    from gtrc import mpl_render_ttb_accuracy as render_ttb
    import SimpleITK as sitk
    if not os.path.exists(render_dir):
        os.mkdir(render_dir)

"""
the input df includes a score for each blob across all of the dataset.
All blobs are scored in terms of basic quantitative metrics (volume, CT#, Mean SUV, etc).
Additionally, each blob's overlap with the (fraction 0.0-1.0) to the expert RT TTB contour is recorded which may be used to
inform the classifier whether to include as a true/false binary decision.
In the first run with sklearn classifiers, I'd set the training data to binary values of 0 or 1 and done a rough weighting
according to the total volume of each blob (repeating it in the training data n times in increments of 10ccs).
In practice, the results of this looked promising but were worse than a very simple majority vote when scored on the
global dice score. I'm now considering whether this is relating to how some partially true labels are weighted as
it doesn't seem logical that a simple majority vote would outperform a more optimised technique.
It would also like to move everything into tensorflow so this attempts to incorporate both improvements.
The challenge is to weight the training data according to the TF dataset mechanics, which it looks like
can be achieved by creating a secondary 'weighting' array which is proportional to the volumetric error (ccs)
if that label was misclassified.

The first processing step is to score the total volume of each blob and calculate the relative difference
in error if misclassified:
eg a label with 1.0 overlap with the expert ttb will be wrong by the total volume of the label whereas
one with true_ttb_overlap=0.9 would result in a net decrease by 80% of the label volume if misclassified

error_fraction = (true_ttb_overlap - (1-ttb_true_overlap)
error_fraction = 2*true_ttb_overlap - 1
error_volume = error_fraction * total_volume
error_volume = abs((2*true_ttb_overlap - 1) * total_volume)

"""

training_columns=['total_volume', 'suv_max', 'suv_mean', 'ct_hu_mean','pred_ttb_overlap','pred_norm_overlap'] #which columns to include for multi-variate model
p=0.2 #dropout rate for consensus model
model = tf.keras.Sequential([ #build small 4-layer fully-connected model
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(p),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(p),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(p),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
csv_logger = CSVLogger(join(fold_dir,'consensus_training_log.csv')) #where to save training logs
model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')],
    weighted_metrics=[tf.keras.losses.categorical_crossentropy]) #build consensus model
checkpointer = ModelCheckpoint(model_path, save_best_only=True, mode='min', monitor='val_loss') #callback to save most accurate model
callbacks=[checkpointer,csv_logger]
df_score["error_volume"]=abs((2*df_score["true_ttb_overlap"]-1) * df_score["total_volume"]) #Compute volumetric error weighting for each region as new column
df_train=df_score[df_score.fold!=(fold_number)] #separate trianing and test rows by fold number
x_train=df_train[training_columns].values #x-data is the training columns designated above
y_train=np.expand_dims(np.round(df_train.iloc[:,8]).astype('int').values,-1) #y-data is the true ttb overlap column with all values >0.5 set to 1 and less set to 0
sample_weight=df_train.error_volume.values #the error is weighted based on the volume of the subregion and the fraction of true ttb overlap

df_val=df_score[df_score.fold==(fold_number)] #same as above for testing/validation data
x_val=df_val[training_columns].iloc[:,:]
y_val=np.expand_dims(np.round(df_val.iloc[:,8]).astype('int').values,-1)
val_weight=df_val.error_volume.values
#run training...
if not skip_consensus_training:
    history = model.fit(x_train, y_train, sample_weight=sample_weight, validation_data=(x_val,y_val,val_weight), epochs=n_epochs,batch_size=32,callbacks=callbacks) 
    print('Consensus Training complete')
else:
    from tensorflow.keras.models import load_model
    model=load_model(model_path,compile=False)

def get_dice(gt,seg): #dice calculation if doing final label analysis
    dice = np.sum(seg[gt==1]==1)*2.0 / (np.sum(seg[seg==1]==1) + np.sum(gt[gt==1]==1))
    return dice

if create_final_labels:
    df_dice=pd.DataFrame(columns=['case','fold','direct_ttb_dice','gtrc_ttb_dice']) #scoring for final and intermediate UNet TTB Dice Scores
    train_dice_scores=[] #just used for screen output
    test_dice_scores=[]
    import SimpleITK as sitk
    if not os.path.exists(final_label_dir): #create directory for post-processed output labels
        os.mkdir(final_label_dir) 
    data_dir=join('data') #top of data directory
    data_csv=join(data_dir,'gtrc_train_data.csv') #summary csv of processed data
    df_data=pd.read_csv(data_csv,index_col=0) #read case input data csv
    ttb_dir=join(data_dir,'ttb_label_adjusted') #input ground truth tumour burden nifti label
    subregion_dir=join(data_dir,'subregions') #to save subregions for each image based on local max and gradient boundary
    for j in range(len(df_data)): #loop through all cases
        case=df_data.iloc[j].case
        fold=df_data.iloc[j].fold
        ws_im=sitk.ReadImage(join(subregion_dir,case)) #load preprocessed subregion image and create numpy array
        ws_ar=sitk.GetArrayFromImage(ws_im) 
        gt_im=sitk.ReadImage(join(ttb_dir,case)) #read in ground truth ttb image for scoring and creaty array
        gt_ar=sitk.GetArrayFromImage(gt_im)
        df_case=df_score[df_score.case==case] #just get rows of subregion dataframe for the current case
        ttb_ar=np.zeros(ws_ar.shape) #create empty array for included subregions
        n_included=0
        try: #try block to fill with zeros if direct ttb labels are missing
            direct_im=sitk.ReadImage(join(fold_dir,'ttb_network_inferred',case)) #if UNet labels are available load and create numpy array
            direct_ar=sitk.GetArrayFromImage(direct_im)
        except:
            direct_ar=np.zeros(ttb_ar.shape)
        for region_num in df_case.region_num.values: #iterate through all subregions
            row=df_case[df_case.region_num==region_num] #get single row with scoring metrics
            x=row[training_columns].values #create consensus model input variable based on training arrays
            pred=model(x).numpy()[0][0] #get prediction [0.0-1.0]
            if pred>=0.5: #if better than 0.5 include subregion
                ttb_ar[ws_ar==region_num]=1
                n_included+=1
        ttb_im=sitk.GetImageFromArray(ttb_ar) #save consensus array to image and copy spatial information
        ttb_im.CopyInformation(ws_im)
        sitk.WriteImage(sitk.Cast(ttb_im,sitk.sitkInt8),join(final_label_dir,case)) #save post-processed labels
        dice=get_dice(gt_ar,ttb_ar) #score post-processed dice
        direct_dice=get_dice(gt_ar,direct_ar) #score low-res UNet dice
        df_dice.loc[j]=[case,fold,direct_dice,dice] #output to case scoring row
        if fold==fold_number: #Just for screen output, append train/test dice scores as appropriate
            test_dice_scores.append(dice)
        else:
            train_dice_scores.append(dice)
        print(j,case,'Dice score:',dice,'Subregions included:',n_included,'/',len(df_case.region_num.values))
        if render_labels:
            qs=sitk.ReadImage(join(qs_dir,case))
            #statistics=render_vtk.render_im_and_struct(qs,gt_im,ttb_im,ptmax=4,output_filename=join(render_dir,case.replace('.nii.gz','png')))
            statistics=render_ttb.create_accuracy_mips(qs,gt_im,ttb_im,ptmax=4,output_filename=join(render_dir,case.replace('.nii.gz','.png')))
            print(statistics)
    print('Training Dice Scores (Ave/Median):',np.average(train_dice_scores),np.median(train_dice_scores))
    print('Validation Dice Scores (Ave/Median):',np.average(test_dice_scores),np.median(test_dice_scores))
    print('All Training is complete. To make a portable model run script #6')
    df_dice.to_csv(join(fold_dir,'final_case_dice_metrics.csv'))



                                     
                                     


