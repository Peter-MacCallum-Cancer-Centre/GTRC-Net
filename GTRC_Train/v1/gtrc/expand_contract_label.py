import SimpleITK as sitk
import numpy as np


def expand_contract_label(label,distance=5.0):
    """expand or contract sitk label image by distance indicated.
    negative values will contract, positive values expand.
    returns sitk image of adjusted label"""
    lar=sitk.GetArrayFromImage(label)
    label_single=sitk.GetImageFromArray((lar>0).astype('int16'))
    label_single.CopyInformation(label)
    distance_filter = sitk.SignedMaurerDistanceMapImageFilter()
    distance_filter.SetUseImageSpacing(True)
    dmap=distance_filter.Execute(label_single)
    dmap_ar=sitk.GetArrayFromImage(dmap)
    new_label_ar=(dmap_ar<=distance).astype('int16')
    new_label=sitk.GetImageFromArray(new_label_ar)
    new_label.CopyInformation(label)
    return new_label


##label=sitk.ReadImage('totseg.nii.gz')
##new_label=expand_contract_label(label,distance=-20)
##sitk.WriteImage(new_label,'expand_contract_test.nii.gz')
