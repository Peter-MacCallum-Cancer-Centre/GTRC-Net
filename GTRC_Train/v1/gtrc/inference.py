import SimpleITK as sitk
import numpy as np
import os
from os.path import join
import pandas as pd



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
