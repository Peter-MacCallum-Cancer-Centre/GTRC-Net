import pandas as pd
import numpy as np
import os,sys
from os.path import join
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint
import argparse
import SimpleITK as sitk


force_training=False

tracer='PSMA' #defaults used if not given command line arguments
fold_number='0'
if len(sys.argv)>2:
    tracer=sys.argv[1]
    fold_number=str(sys.argv[2])
    print('tracer:',tracer)
    print('fold_number:',fold_number)


fold_dir=join('data',tracer,'subregion_metrics','fold_'+str(fold_number))
    
df_score=pd.read_csv(join(fold_dir,'subregion_prediction_metrics.csv'),index_col=0) #read in overlap and scoring metrics from previous script

n_epochs=1000 #number of epochs to run consensus model
create_final_labels=True #whether to save final prediction labels and score dice metrics
render_labels=False
final_label_dir=join(fold_dir,'final_labels')
error_label_dir=join(fold_dir,'error_labels')
nn_error_label_dir=join(fold_dir,'nn_error_labels')

data_dir=join('data',tracer) #top of data directory
data_csv=join(data_dir,'validation_folds.csv') #summary csv of processed data
df_data=pd.read_csv(data_csv,index_col=0) #read case input data csv
ttb_dir=join(data_dir,'TTB') #input ground truth tumour burden nifti label
norm_dir=join(data_dir,'normal')
pt_dir=join(data_dir,'PET_rescaled') #qspect output after scaling intensity to value of 1.0 at min of detected TTB per case


model_path=join(fold_dir,'consensus_model.keras')
print(model_path)

if os.path.exists(model_path) and not force_training:
    print('Existing consensus model found:',model_path)
    print('Back up to new location or re-run with force training flag (--force/-f) train updated model')
    print('proceeding with inference and final statistics...')
    skip_consensus_training=True
else:
    skip_consensus_training=False

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
    weighted_metrics=[tf.keras.losses.binary_crossentropy]) #build consensus model    

checkpointer = ModelCheckpoint(model_path, save_best_only=True, mode='min', monitor='val_loss') #callback to save most accurate model
callbacks=[checkpointer,csv_logger]
df_score["error_volume"]=abs((2*df_score["true_ttb_overlap"]-1) * df_score["total_volume"]) #Compute volumetric error weighting for each region as new column
df_train=df_score[df_score.fold!=(int(fold_number))] #separate trianing and test rows by fold number

x_train=df_train[training_columns].values #x-data is the training columns designated above
y_train=np.expand_dims(np.round(df_train.iloc[:,8]).astype('int').values,-1) #y-data is the true ttb overlap column with all values >0.5 set to 1 and less set to 0
sample_weight=df_train.error_volume.values #the error is weighted based on the volume of the subregion and the fraction of true ttb overlap

df_val=df_score[df_score.fold==(int(fold_number))] #same as above for testing/validation data
print('n train/val cases',len(df_train),len(df_val))
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

rs=sitk.ResampleImageFilter()
rs.SetInterpolator(sitk.sitkNearestNeighbor)
if create_final_labels:
    import SimpleITK as sitk
    df_dice=pd.DataFrame(columns=['case','fold','direct_ttb_dice','gtrc_og_dice','gtrc_ttb_dice']) #,'nn_fold'scoring for final and intermediate UNet TTB Dice Scores
    train_dice_scores=[] #just used for screen output
    test_dice_scores=[]
    os.makedirs(final_label_dir,exist_ok=True)
    os.makedirs(error_label_dir,exist_ok=True)
    os.makedirs(nn_error_label_dir,exist_ok=True)
    
    subregion_dir=join(data_dir,'subregions') #to save subregions for each image based on local max and gradient boundary
    for j in range(len(df_data)): #loop through all cases
        case=df_data.iloc[j].case
        fold=df_data.iloc[j].val_fold
        ws_im=sitk.ReadImage(join(subregion_dir,case)) #load preprocessed subregion image and create numpy array
        ws_ar=sitk.GetArrayFromImage(ws_im) 
        gt_im=sitk.ReadImage(join(ttb_dir,case)) #read in ground truth ttb image for scoring and creaty array
        norm_im=sitk.ReadImage(join(norm_dir,case))
        norm_ar=sitk.GetArrayFromImage(norm_im)
        pt=sitk.ReadImage(join(pt_dir,case))
        pt_ar=sitk.GetArrayFromImage(pt)
        gt_ar=sitk.GetArrayFromImage(gt_im)
        df_case=df_score[df_score.case==case] #just get rows of subregion dataframe for the current case
        ttb_ar=np.zeros(ws_ar.shape) #create empty array for included subregions
        inf_norm_ar=np.zeros(ws_ar.shape)
        n_included=0
        try: #try block to fill with zeros if direct ttb labels are missing

            inferred_ttb_dir=join('data','nn_inferred',tracer,'fold_'+str(fold_number))
            direct_im=sitk.ReadImage(join(inferred_ttb_dir,case)) #if UNet labels are available load and create numpy array
            direct_ar=sitk.GetArrayFromImage(direct_im)
            direct_ar=(direct_ar==1).astype('int8') #ignore background label (2)s
        except Exception as e:
            print(case,e)
            direct_ar=np.zeros(ttb_ar.shape)
        for region_num in df_case.region_num.values: #iterate through all subregions
            row=df_case[df_case.region_num==region_num] #get single row with scoring metrics
            x=row[training_columns].values #create consensus model input variable based on training arrays
            pred=model(x).numpy()[0][0] #get prediction [0.0-1.0]
            if pred>=0.5: #if better than 0.5 include subregion
                ttb_ar[ws_ar==region_num]=1
                n_included+=1
            else:
                inf_norm_ar[ws_ar==region_num]=1

        #to include direct nnUNet-predicted voxels after consensus:
        gtrc_og_dice=get_dice(gt_ar,ttb_ar)
        
        ttb_ar[direct_ar==1]=1
        ttb_ar[pt_ar<1.0]=0
        
        ttb_im=sitk.GetImageFromArray(ttb_ar) #save consensus array to image and copy spatial information
        ttb_im.CopyInformation(ws_im)
        sitk.WriteImage(sitk.Cast(ttb_im,sitk.sitkInt8),join(final_label_dir,case+'.nii.gz')) #save post-processed labels

        dice=get_dice(gt_ar,ttb_ar) #score post-processed dice
        direct_dice=get_dice(gt_ar,direct_ar) #score low-res UNet dice
        df_dice.loc[j]=[case,fold,direct_dice,gtrc_og_dice,dice]#,nn_fold] #output to case scoring row

        if fold==fold_number: #Just for screen output, append train/test dice scores as appropriate
            test_dice_scores.append(dice)
        else:
            train_dice_scores.append(dice)
        print(j,case,'Dice scores (direct/GTRC OG/GTRC NN TTB Re-added):',direct_dice,gtrc_og_dice,dice,'\nSubregions included:',n_included,'/',len(df_case.region_num.values))

    print('All Training is complete.')
    df_dice.to_csv(join(fold_dir,'final_case_dice_metrics.csv'))



                                     
                                     


