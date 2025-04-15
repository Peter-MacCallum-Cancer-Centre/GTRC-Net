import SimpleITK as sitk
import numpy as np
from numpy.random import rand
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
import os
from os.path import join
import pandas as pd
import random  #add to imports

def match_array_to_image(ar,ref):
    im=sitk.GetImageFromArray(ar)
    im.SetOrigin(ref.GetOrigin())
    im.SetSpacing(ref.GetSpacing())
    im.SetDirection(ref.GetDirection())
    return im

def random_crop(image1,label1,image2=False,label2=False):
    ar1=sitk.GetArrayFromImage(image1)
    shape=ar1.shape
    label_sum=0
    min_slices=20
    n_iterations=1
    cropfilter=sitk.CropImageFilter()
    normal_factor=4
    while label_sum==0:
        if True: #whether to start from top
            t_crop=int(abs(np.random.normal(0,shape[0]/normal_factor)))
            bt_crop=int(abs(np.random.normal(0,shape[0]/normal_factor)))
            if t_crop+bt_crop>(shape[0]-min_slices):
                label_sum=0 #restart if not enough slices potentially...
                n_iterations+=1
            else:
                l_crop=int(abs(np.random.normal(0,shape[2]/normal_factor)))
                r_crop=int(abs(np.random.normal(0,shape[2]/normal_factor)))
                if l_crop+r_crop>(shape[2]-min_slices):
                    label_sum=0
                    n_iterations+=1
                else:
                    f_crop=int(abs(np.random.normal(0,shape[1]/normal_factor)))
                    bk_crop=int(abs(np.random.normal(0,shape[1]/normal_factor)))
                    if f_crop+bk_crop>(shape[1]-min_slices):
                        label_sum=0
                        n_iterations+=1
                    else:
                        cropfilter.SetLowerBoundaryCropSize([r_crop,bk_crop, bt_crop])
                        cropfilter.SetUpperBoundaryCropSize([l_crop,f_crop, t_crop])
                        lar1_crop=sitk.GetArrayFromImage(cropfilter.Execute(label1))
                        if label2:
                            lar2_crop=sitk.GetArrayFromImage(cropfilter.Execute(label2))
                            if lar1_crop.sum()>0 and lar2_crop.sum()>0:
                                #print('2x labels, crop extents',[r_crop,bk_crop, bt_crop],[l_crop,f_crop, t_crop],n_iterations,lar1_crop.sum(),lar2_crop.sum(),lar1_crop.size,lar2_crop.size)
                                label_sum=1
                        else:
                            if lar1_crop.sum()>0:
                                #print('1x label, crop extents',[r_crop,bk_crop, bt_crop],[l_crop,f_crop, t_crop],n_iterations,lar1_crop.sum(),lar1_crop.size)
                                label_sum=1
    cropfilter.SetLowerBoundaryCropSize([r_crop,bk_crop, bt_crop])
    cropfilter.SetUpperBoundaryCropSize([l_crop,f_crop, t_crop])
    image1_crop=cropfilter.Execute(image1)
    label1_crop=cropfilter.Execute(label1)
    if image2:
        image2_crop=cropfilter.Execute(image2)
        if label2:
            label2_crop=cropfilter.Execute(label2)
            return image1_crop, image2_crop, label1_crop, label2_crop
        else:
            return image1_crop, image2_crop, label1_crop
    else:
        if label2:
            label2_crop=cropfilter.Execute(label2)
            return image1_crop, label1_crop, label2_crop
        else:
            return image1_crop, label1_crop


def load_cropped(resample_extent,resample_dimensions,qs_path,ttb_path=False,ct_path=False,norm_path=False,augment=True,
                 max_rotation_deg=20.,max_translation=30.,max_gauss_sigma=1.0,
                 max_hu_shift=30,max_noise=100,sharpening_range=0.6,sharpening_alpha=0.5,qs_scaler=0.05,crop_augment=True,include_background_channel=True,batch_shape=False):
    if ttb_path==False and norm_path==False:
        print('Need either ttb or normal label')
    if ttb_path and norm_path:
        both=True
    else:
        both=False
    np.random.seed(random.randint(0,65535)) #add at start of function before random augment calls
    zeds=np.zeros(resample_dimensions)
    rss=np.array((resample_extent[0]/resample_dimensions[2],resample_extent[1]/resample_dimensions[1],resample_extent[2]/resample_dimensions[0])) #sitk xyz
    qs=sitk.Cast(sitk.ReadImage(qs_path),sitk.sitkFloat32)
    if ttb_path:
        ttb=sitk.ReadImage(ttb_path)
    rs=sitk.ResampleImageFilter()
    rs.SetInterpolator(sitk.sitkNearestNeighbor)
    rs.SetDefaultPixelValue(0)
    rs.SetReferenceImage(qs)
    if ttb_path:
        ttb=sitk.Cast(rs.Execute(ttb),sitk.sitkInt16)
    if norm_path:
        norm=sitk.ReadImage(norm_path)
        norm=sitk.Cast(rs.Execute(norm),sitk.sitkInt16)
    if ct_path:
        ct=sitk.ReadImage(ct_path)
        rs.SetDefaultPixelValue(-1000)
        rs.SetInterpolator(sitk.sitkLinear)
        ct=sitk.Cast(rs.Execute(ct),sitk.sitkFloat32)

    if crop_augment and augment:
        if ct_path:
            if both:
                qs,ct,ttb,norm=random_crop(qs,ttb,image2=ct,label2=norm)
            elif ttb_path:
                qs,ct,ttb=random_crop(qs,ttb,image2=ct)
            else:
                qs,ct,norm=random_crop(qs,norm,image2=ct)
        else:
            if both:
                qs,ttb,norm=random_crop(qs,ttb,label2=norm)
            elif ttb_path:
                qs,ttb=random_crop(qs,ttb)
            else:
                qs,norm=random_crop(qs,norm)            
    origin=np.array(qs.GetOrigin())
    original_dims=np.array(qs.GetSize())
    original_spacing=np.array(qs.GetSpacing())
    original_extent=original_dims*original_spacing
    origin_shift=(0.5)*(resample_extent[2]-original_extent[2]) #puts in centre of slab
    origin[2]=origin[2]-origin_shift
    delta_extent=resample_extent-original_extent
    delta_x=delta_extent[0]/2.
    delta_y=delta_extent[1]/2.
    new_origin=np.array((origin[0]-delta_x,origin[1]-delta_y,origin[2]))
    ref=sitk.GetImageFromArray(zeds)
    ref.SetSpacing(rss)
    ref.SetOrigin(new_origin)    
    if augment:
        sig=np.random.rand()*max_gauss_sigma
        sharp=sharpening_range*(1-0.3*(np.random.rand()-0.5))
        salpha=sharpening_alpha*(1-0.8*(np.random.rand()-0.5))
        hu_shift=(np.random.rand()-0.5)*max_hu_shift
        sig=np.random.rand()*max_gauss_sigma
        sharp=sharpening_range*(1-0.3*(np.random.rand()-0.5))
        salpha=sharpening_alpha*(1-0.8*(np.random.rand()-0.5))
        hu_shift=(np.random.rand()-0.5)*max_hu_shift
        rotx=(np.random.rand()-0.5)*max_rotation_deg*2*3.1415/360
        roty=(np.random.rand()-0.5)*max_rotation_deg*2*3.1415/360
        rotz=(np.random.rand()-0.5)*max_rotation_deg*2*3.1415/360
        tx=(np.random.rand()-0.5)*max_translation
        ty=(np.random.rand()-0.5)*max_translation
        tz=(np.random.rand()-0.5)*max_translation
        initial_transform=sitk.Euler3DTransform()
        initial_transform.SetParameters((rotx,roty,rotz,tx,ty,tz))
        qs = sitk.Resample(qs, ref, initial_transform, sitk.sitkLinear, 0., sitk.sitkFloat32)
        qs_ar=sitk.GetArrayFromImage(qs)
        qs_mult=np.random.normal(1.0,qs_scaler)
        qs_ar=qs_ar*qs_mult #just hard-coded as a quick one, ranges about 0.3 to 3.0
        if ttb_path:
            ttb=sitk.Resample(ttb,ref,initial_transform,sitk.sitkNearestNeighbor, 0.0, sitk.sitkUInt8)
            ttb_ar=sitk.GetArrayFromImage(ttb)
        if norm_path:
            norm=sitk.Resample(norm,ref,initial_transform,sitk.sitkNearestNeighbor, 0.0, sitk.sitkUInt8)
            norm_ar=sitk.GetArrayFromImage(norm)
        if ct_path:
            ct = sitk.Resample(ct, ref, initial_transform, sitk.sitkLinear, -1000., sitk.sitkFloat32)
            ct_ar=sitk.GetArrayFromImage(ct)
            blurred_ar=gaussian_filter(ct_ar,sharp)
            sharpened=ct_ar+salpha*(ct_ar-blurred_ar)
            ct_ar=sharpened
            ct_ar=gaussian_filter(ct_ar,sigma=sig)
            ct_ar+=int(hu_shift)
            ct_ar+=((np.random.random(ct_ar.shape)-0.5)*max_noise).astype('int16')
    else:
        rs.SetReferenceImage(ref)
        rs.SetInterpolator(sitk.sitkLinear)
        rs.SetDefaultPixelValue(0)
        qs=rs.Execute(qs)
        qs_ar=sitk.GetArrayFromImage(qs)
        if ct_path:
            rs.SetDefaultPixelValue(-1000)
            ct=rs.Execute(ct)
            ct_ar=sitk.GetArrayFromImage(ct)
        rs.SetInterpolator(sitk.sitkNearestNeighbor)
        rs.SetDefaultPixelValue(0)
        if ttb_path:
            ttb=rs.Execute(ttb)
            ttb_ar=sitk.GetArrayFromImage(ttb)
        if norm_path:
            norm=rs.Execute(norm)
            norm_ar=sitk.GetArrayFromImage(norm)
    x=np.expand_dims(qs_ar,-1)
    if ct_path:
        x=np.append(x,np.expand_dims(ct_ar,-1),axis=-1)
    
    if ttb_path and norm_path:
        y=np.append(np.expand_dims(ttb_ar,-1),np.expand_dims(norm_ar,-1),axis=-1)
    elif ttb_path:
        y=np.expand_dims(ttb_ar,-1)
    else:
        y=np.expand_dims(norm_ar,-1)
    if include_background_channel:
        labelled=np.expand_dims(np.sum(y,axis=-1),-1)
        y=np.append((labelled==0),y,axis=-1)
    if not batch_shape: #omit first axis for later stacking in tf dataset batch
        x=np.expand_dims(x,0)
        y=np.expand_dims(y,0)
    return x,y

def load_cropped_inference(resample_extent,resample_dimensions,qs_path,ct_path=False,batch_shape=False):
    zeds=np.zeros(resample_dimensions)
    rss=np.array((resample_extent[0]/resample_dimensions[2],resample_extent[1]/resample_dimensions[1],resample_extent[2]/resample_dimensions[0])) #sitk xyz
    qs=sitk.Cast(sitk.ReadImage(qs_path),sitk.sitkFloat32)
    rs=sitk.ResampleImageFilter()
    rs.SetReferenceImage(qs)
    if ct_path:
        ct=sitk.ReadImage(ct_path)
        rs.SetDefaultPixelValue(-1000)
        rs.SetInterpolator(sitk.sitkLinear)
        ct=sitk.Cast(rs.Execute(ct),sitk.sitkFloat32)
    origin=np.array(qs.GetOrigin())
    original_dims=np.array(qs.GetSize())
    original_spacing=np.array(qs.GetSpacing())
    original_extent=original_dims*original_spacing
    origin_shift=(0.5)*(resample_extent[2]-original_extent[2]) #puts in centre of slab
    origin[2]=origin[2]-origin_shift
    delta_extent=resample_extent-original_extent
    delta_x=delta_extent[0]/2.
    delta_y=delta_extent[1]/2.
    new_origin=np.array((origin[0]-delta_x,origin[1]-delta_y,origin[2]))
    ref=sitk.GetImageFromArray(zeds)
    ref.SetSpacing(rss)
    ref.SetOrigin(new_origin)    


    rs.SetReferenceImage(ref)
    rs.SetInterpolator(sitk.sitkLinear)
    rs.SetDefaultPixelValue(0)
    qs=rs.Execute(qs)
    qs_ar=sitk.GetArrayFromImage(qs)
    if ct_path:
        rs.SetDefaultPixelValue(-1000)
        ct=rs.Execute(ct)
        ct_ar=sitk.GetArrayFromImage(ct)
           
    x=np.expand_dims(qs_ar,-1)
    if ct_path:
        x=np.append(x,np.expand_dims(ct_ar,-1),axis=-1)

    return x,qs

##def load_training_case(fname,resample_extent,resample_dimensions,qs_dir,ttb_dir,norm_dir,ct_dir):
##    fname=fname.numpy().decode('utf-8')
##    qs_path=join(qs_dir,fname)
##    if include_ct:
##        ct_path=join(ct_dir,fname)
##    else:
##        ct_path=False
##    ttb_path=join(ttb_dir,fname)
##    norm_path=join(norm_dir,fname)
##    if region=='tumor':
##        norm_path=False
##    if region=='normal':
##        ttb_path=False
##    x,y=load_ttb.load_cropped(resample_extent, resample_dimensions, qs_path, ttb_path, ct_path,norm_path, augment=True,
##                              include_background_channel=True,batch_shape=True,crop_augment=True)
##    return x.astype('float32'),y.astype('float32')
##
##def load_testing_case(fname,resample_extent,resample_dimensions,qs_dir,ttb_dir,norm_dir,ct_dir):
##    fname=fname.numpy().decode('utf-8')
##    qs_path=join(qs_dir,fname)
##    if include_ct:
##        ct_path=join(ct_dir,fname)
##    else:
##        ct_path=False
##    ttb_path=join(ttb_dir,fname)
##    norm_path=join(norm_dir,fname)
##    if region=='tumor':
##        norm_path=False
##    if region=='normal':
##        ttb_path=False
##    x,y=load_ttb.load_cropped(resample_extent, resample_dimensions, qs_path, ttb_path, ct_path,norm_path, augment=False,include_background_channel=include_background_channel,batch_shape=True)
##    return x.astype('float32'),y.astype('float32') 
