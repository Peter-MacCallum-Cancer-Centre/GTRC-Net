import os
from os.path import join
import numpy as np
import SimpleITK as sitk
import random
import pandas as pd

from gtrc.gtrc_utils import plot_mip
from gtrc.gtrc_utils import create_subregion_labels
from gtrc.expand_contract_label import expand_contract_label

"""training data should be in nifti (.nii.gz) or other ITK-readible format
Images should be PET/CT or QSPECT/CT with tumour burden labels; 3 nifti files per case.
Each case should have the same file name (eg Patient00001.nii.gz) in each data folder:
data/ct/Patient00001.nii.gz
data/qspect/Patient00001.nii.gz
data/ttb_label/Patient00001.nii.gz
Depending on how the images have been converted from dicom format, the intensity values
of the functional (PET/QSPECT) images may be in either Bq/ml or SUV, however for the purposes
of data pre-processing we will rescale the qspect images according to the minimum detected value
in the TTB contour label. This creates a rescaled functional nifti for each case such that the
appropriate threshold should be at a value of 1.0.
CT images are also pre-processed to match the resolution of the PET/QSPECT images which
permits faster file-loading from storage.
Lastly, an inferred physiological/normal uptake label set is taken from the boolean
subtraction of the values above the threshold (>=1.0) and the expert tumor burden contour.
The final training data will be generated into the folders:
data/ct_lowres/*.nii.gz
data/qspect_scaled/*.nii.gz
data/ttb_label/*.nii.gz
data/norm_label/.nii.gz
The preprocessing will generate a set of maximum-intensity projection screenshots
which may be used to review the accuracy of processing stages (data/preprocessed_mips)
and where necessary facilitate correction or removal of erroneous cases.
Lastly a summary .csv is generated which lists all of the cases, assigns a fold #, and
logs a handful of measurements based on the image sets
data/preprocessed_training_cases.csv
Note: after preprocessing the original CT and qspect nifti images may be removed if needed
to reduce storage




"""


data_dir=join('data') #top of data directory
ct_in=join(data_dir,'ct') #ct input nifti location
ct_out=join(data_dir,'ct_lowres') #CT output directory matched to QSPECT resolution
qs_in=join(data_dir,'qspect') #qspect input nifti location
qs_out=join(data_dir,'qspect_rescaled') #qspect output after scaling intensity to value of 1.0 at min of detected TTB per case
ttb_dir=join(data_dir,'ttb_label') #input ground truth tumour burden nifti label
ttb_adjusted_dir=join(data_dir,'ttb_label_adjusted')
norm_dir=join(data_dir,'norm_label') #output derived normal/physiological nifti - above min threshold but not included in TTB
mip_dir=join(data_dir,'preprocessed_mips') #output folder for quick review of accuracy of scaled images and derived normal labels
subregion_dir=join(data_dir,'subregions') #to save subregions for each image based on local max and gradient boundary
extension='.nii.gz' #file extension of input data
data_csv_out=join(data_dir,'gtrc_train_data.csv') #summary csv of processed data

##local_maxima_threshold=0.16667 #local max needed for new subregion with respect to neighbouring area. Smaller value should be more granular
##sphere_radius=6 #in mm, radius from possible nearest adjacent subregion. Smaller value should be more granular

local_maxima_threshold=0.25 #local max needed for new subregion with respect to neighbouring area. Smaller value should be more granular
sphere_radius=8 #in mm, radius from possible nearest adjacent subregion. Smaller value should be more granular
#label_expansion_radius=4.0
label_expansion_radius=60. ####updated from 4, 20 looks pretty good
min_bq_thresh=2.90

if not os.path.exists(ct_out):
    os.mkdir(ct_out)
if not os.path.exists(qs_out):
    os.mkdir(qs_out)
if not os.path.exists(norm_dir):
    os.mkdir(norm_dir)
if not os.path.exists(mip_dir):
    os.mkdir(mip_dir)
if not os.path.exists(subregion_dir):
    os.mkdir(subregion_dir)
if not os.path.exists(ttb_adjusted_dir):
    os.mkdir(ttb_adjusted_dir)

n_folds=5 #number of folds to prepare, will be saved into a csv
df=pd.DataFrame(columns=['case','fold','min_ttb_intensity','ttb_volume_cc',
                         'norm_volume_cc','resolution','spacing','extent',
                         'subregion_local_max','subregion_radius','n_subregions'])

rs=sitk.ResampleImageFilter() #sitk resampler object
suv_threshold=1.0 #threshold value to rescale all qspect images 
#np_percentile=0.1 #near-zero percentile value to use for detecting minimum/threshold in ttb contours
np_percentile=0.0 #near-zero percentile value to use for detecting minimum/threshold in ttb contours

cases=os.listdir(ct_in) #list all images in input directory, will test if present in qspect & ttb as well
random.seed(1) #random seed for shuffling training data and selecting folds
random.shuffle(cases)
counter=1
for case in cases:
    if case in os.listdir(qs_in) and case in os.listdir(ttb_dir): #check if all image types present
        include=True
        label=sitk.ReadImage(join(ttb_dir,case)) #read ttb label
        ct=sitk.ReadImage(join(ct_in,case)) #read ct
        qs=sitk.ReadImage(join(qs_in,case)) #read qspect
        rs.SetReferenceImage(qs) #resample label to qspect spacing and array dimensions (sometimes smaller array if converted via plastimatch)
        rs.SetInterpolator(sitk.sitkNearestNeighbor)
        rs.SetDefaultPixelValue(0)
        label=rs.Execute(label)
        lar=sitk.GetArrayFromImage(label)
        qar=sitk.GetArrayFromImage(qs) #get numpy arrays for images
        bq_thresh=np.percentile(qar[lar>0],np_percentile)
        print(bq_thresh)
        if min_bq_thresh:
            if bq_thresh<min_bq_thresh:
                bq_thresh=min_bq_thresh

        #### Addition label dilation before rethreshold...
        label=expand_contract_label(label,label_expansion_radius)
        
        #### Addition label dilation before rethreshold...
        
        rs.SetInterpolator(sitk.sitkLinear) #resample CT to match qspect resolution, makes training more efficient
        rs.SetDefaultPixelValue(-1000)
        ct_lowres=rs.Execute(ct)
        voxel_volume=np.prod(np.array(label.GetSpacing()))/1000. #for calculating ttb/normal label volumes
        
        lar=sitk.GetArrayFromImage(label)
        lar=np.logical_and(lar,qar>=bq_thresh)
        ttb_adjusted_label=sitk.GetImageFromArray(lar.astype('int16'))
        ttb_adjusted_label.CopyInformation(label)
        sitk.WriteImage(ttb_adjusted_label,join(ttb_adjusted_dir,case))
##        bq_thresh=np.percentile(qar[lar>0],np_percentile) #determine minimum (~0th percentile) value included in TTB contour as assumed global threshold
        suv_factor=(bq_thresh/suv_threshold)
        suv_ar=qar/suv_factor #rescale intensity of qspect image data
        threshold_ar=suv_ar>=suv_threshold #get all image volume above global threshold
        norm_lar=np.logical_and(lar==0,threshold_ar) #boolean subtraction for normal uptake (above thresh but not in ttb)
        if norm_lar.sum()==0:
            print('No Normal voxels detected. Not including case in training...')
            include=False
        norm_label=sitk.GetImageFromArray(norm_lar.astype('int16')) #create and write out derived normal label
        norm_label.CopyInformation(label)
        sitk.WriteImage(norm_label,join(norm_dir,case)) 
        qs_suv=sitk.GetImageFromArray(suv_ar) #write out rescaled qspect image
        qs_suv.CopyInformation(qs)
        sitk.WriteImage(qs_suv,join(qs_out,case))
        sitk.WriteImage(ct_lowres,(join(ct_out,case))) #write out low res ct image
        fold=(counter%n_folds)+1 #to iterate fold numbers
        norm_volume_cc=norm_lar.sum()*voxel_volume #total volume in ccs of normal label
        ttb_volume_cc=lar.sum()*voxel_volume #total volume in ccs ttb label
        resolution=qs.GetSize()
        spacing=qs.GetSpacing()
        extent=(np.array(resolution)*np.array(spacing)).tolist()
        subregion_image=create_subregion_labels(qs_suv,threshold_ar,local_maxima_threshold,sphere_radius)
        n_subregions=sitk.GetArrayFromImage(subregion_image).max()
        sitk.WriteImage(subregion_image,join(subregion_dir,case))
##        n_subregions=0
        print(counter,case,fold,bq_thresh,ttb_volume_cc,norm_volume_cc,resolution,spacing,extent,n_subregions)
        if include:
            df.loc[counter]=[case,fold,bq_thresh,ttb_volume_cc,norm_volume_cc,resolution,
                             spacing,extent,local_maxima_threshold,sphere_radius,n_subregions] #write to csv
            counter+=1
        #create 2-view MIP (cor/sag) of rescaled qspect overlaid with ttb & normal labels for quick review
        plot_mip(qs_suv,ttb_adjusted_label,norm_label,join(mip_dir,case.replace(extension,'.png')),case,show=False)
    else:
        print(case,'must be located in all three input data folders: ct, qspect, & ttb_label')
df.to_csv(data_csv_out) #save csv which will be referenced for all future file loading.
#data can be omitted from training by removing rows from this csv (for example if error seen in review of MIP images)
