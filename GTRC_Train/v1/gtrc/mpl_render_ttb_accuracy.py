import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join
import SimpleITK as sitk


def get_dice(gt,seg):
    dice = np.sum(seg[gt==1]==1)*2.0 / (np.sum(seg[seg==1]==1) + np.sum(gt[gt==1]==1))
    return dice

def create_accuracy_mips(qspect,struct_gt,struct_pred,ptmax=30,output_filename='label_accuracy.png'):
    qs_ar=sitk.GetArrayFromImage(qspect)
    rss=qspect.GetSpacing()
    voxel_volume=np.prod(np.array(struct_gt.GetSpacing()))/1000. #volume in cc
    ttb_ar=sitk.GetArrayFromImage(struct_gt)
    svm_cleaned_ar=sitk.GetArrayFromImage(struct_pred)
    dice_score=round(get_dice(ttb_ar,svm_cleaned_ar),4)            
    true_pos_ar=np.logical_and((svm_cleaned_ar>0),(ttb_ar>0.5))
    false_pos_ar=np.logical_and((svm_cleaned_ar>0),(ttb_ar<0.5))
    false_neg_ar=np.logical_and((svm_cleaned_ar==0),(ttb_ar>0.5))
    ttb_volume=voxel_volume*ttb_ar.sum()
    #'index','case','fold','dice','true_pos','false_pos','false_neg'
    true_pos_vol=round(true_pos_ar.sum()*voxel_volume,1)
    false_pos_vol=round(false_pos_ar.sum()*voxel_volume,1)
    false_neg_vol=round(false_neg_ar.sum()*voxel_volume,1)
    vol_gt=round((ttb_ar>0).sum()*voxel_volume,1)
    vol_pred=round((svm_cleaned_ar>0).sum()*voxel_volume,1)
    mean_gt=round(qs_ar[ttb_ar>0].mean(),2)
    mean_pred=round(qs_ar[svm_cleaned_ar>0].mean(),2)    

    plt.figure(figsize=[12,12])
    plt.subplot(121)
    linewidth=1.0
    plt.imshow(np.flipud(np.amax(qs_ar,axis=1)),cmap='Greys',aspect=(rss[2]/rss[0]),clim=[0,ptmax])
    plt.contourf(np.flipud(np.amax(svm_cleaned_ar,axis=1)),levels=[0.5,1.5],colors='g',alpha=0.5) #linewidths=linewidth
    plt.contourf(np.flipud(np.amax(false_pos_ar,axis=1)),levels=[0.5,1.5],colors='r',alpha=0.5)
    plt.contourf(np.flipud(np.amax(false_neg_ar,axis=1)),levels=[0.5,1.5],colors='b',alpha=0.5)
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(np.flipud(np.amax(qs_ar,axis=2)),cmap='Greys',aspect=(rss[2]/rss[0]),clim=[0,ptmax])
    #plt.imshow(np.flipud(np.average(ct_ar,axis=2)),cmap='Greys_r',aspect=(rss[2]/rss[1]))
    plt.contourf(np.flipud(np.amax(true_pos_ar,axis=2)),levels=[0.5,1.5],colors='g',alpha=0.5)
    plt.contourf(np.flipud(np.amax(false_pos_ar,axis=2)),levels=[0.5,1.5],colors='r',alpha=0.5)
    plt.contourf(np.flipud(np.amax(false_neg_ar,axis=2)),levels=[0.5,1.5],colors='b',alpha=0.5)    
    #plt.contour(np.flipud(np.amax(ar,axis=2)),levels=[0.5],colors=color2)
    plt.axis('off')

    plt.suptitle(output_filename.split('.')[0]+' '+str(ttb_ar.shape[1])+' '+str(dice_score)+'\nG: True+,  R: False+,  B: False-')
    plt.tight_layout()
    plt.savefig(output_filename)
    statistics=[dice_score,true_pos_vol,false_pos_vol,false_neg_vol,vol_gt,vol_pred,mean_gt,mean_pred]
    return statistics


