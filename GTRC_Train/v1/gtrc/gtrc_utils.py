import os
from os.path import join
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pandas as pd
from skimage import morphology
from scipy import ndimage
from skimage.segmentation import watershed

def plot_mip(qspect,lab_ttb,lab_norm,outpath,title='MIP View - TTB & Normal',clim_max=2.5,show=False):
    nar=sitk.GetArrayFromImage(lab_norm)
    tar=sitk.GetArrayFromImage(lab_ttb)    
    plt.figure(figsize=[12,6])
    spacing=qspect.GetSpacing()
    aspect=spacing[2]/spacing[0]    
    ar=sitk.GetArrayFromImage(qspect)
    plt.subplot(121)
    plt.imshow(np.flipud(np.amax(ar,1)),aspect=aspect,cmap='Greys',clim=[0,clim_max])
    
    plt.contour(np.flipud(np.amax(tar,1)),colors='r',levels=[0.5],linewidths=0.8)
    plt.contourf(np.flipud(np.amax(nar,1)),colors='b',levels=[0.5,1.5],linewidths=0.8,alpha=0.7)
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(np.flipud(np.amax(ar,2)),aspect=aspect,cmap='Greys',clim=[0,clim_max])
    
    plt.contour(np.flipud(np.amax(tar,2)),colors='r',levels=[0.5],linewidths=0.8)
    plt.contourf(np.flipud(np.amax(nar,2)),colors='b',levels=[0.5,1.5],linewidths=0.8,alpha=0.7)
    plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(outpath)
    if show:
        plt.show()
    plt.close('all')
    return

def get_expansion_sphere(radius,spacing):
    xlim=np.ceil(radius/spacing[0])
    ylim=np.ceil(radius/spacing[1])
    zlim=np.ceil(radius/spacing[2])
    x=np.arange(-xlim,xlim+1,1)
    y=np.arange(-ylim,ylim+1,1)
    z=np.arange(-zlim,zlim+1,1)
    xx,yy,zz=np.meshgrid(x,y,z)
    sphere=(np.sqrt((xx*spacing[0])**2+(yy*spacing[1])**2+(zz*spacing[2])**2)<=radius).astype(np.float32)
    sphere=np.swapaxes(sphere,0,2)
    return sphere

def create_subregion_labels(qspect,threshold_array,local_maxima_threshold,sphere_radius):
    qs_ar=sitk.GetArrayFromImage(qspect)
    spacing=qspect.GetSpacing()
    sphere=get_expansion_sphere(sphere_radius,spacing)
    peaks=morphology.h_maxima(qs_ar,local_maxima_threshold,sphere)
    numbered_peaks,n_peaks=ndimage.label(peaks)
    labels_ws=watershed(-qs_ar,numbered_peaks,mask=threshold_array)
    missing_peaks_ar=threshold_array-(labels_ws>0).astype('uint8') #some non-contiguous regions above threshold may not meet criteria for detection by local max but should still be included
    if missing_peaks_ar.sum()>0:
        missing_peaks_ar_numbered,n_missing=ndimage.label(missing_peaks_ar)
        for i in range(n_missing): #catch to include regions that don't get identified by h_maxima
            labels_ws[missing_peaks_ar_numbered==(i+1)]=n_peaks+i
    ws_im=sitk.GetImageFromArray(labels_ws)
    ws_im.CopyInformation(qspect)
    ws_im=sitk.Cast(ws_im,sitk.sitkInt16)
    return ws_im
