import vtk
import SimpleITK as sitk
import numpy as np
import os
from os.path import join
#from load_cropped_case_v1_render import load_cropped
import pandas as pd
import os, sys, subprocess
#from PIL import Image
from vtk.util import numpy_support
import numpy as np
from scipy import ndimage
from scipy.ndimage import sobel

from vtkmodules.vtkCommonCore import (
    vtkLookupTable)
from vtkmodules.vtkCommonColor import (
    vtkColorSeries,
    vtkNamedColors
)
from vtkmodules.vtkRenderingCore import (
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer,
    vtkTextActor
)

def match_array_to_im(ar,ref):
    out=sitk.GetImageFromArray(ar)
    out.SetOrigin(ref.GetOrigin())
    out.SetSpacing(ref.GetSpacing())
    out.SetDirection(ref.GetDirection())
    return out


epoch=1
colors = vtk.vtkNamedColors()
def resample_cubic(im,output_spacing):
    spacing=np.array(im.GetSpacing())
    size=np.array(im.GetSize())
    extent=spacing*size
    dims=np.ceil(extent/output_spacing).astype('int16')
    zeds=np.zeros(dims)
    ref=sitk.GetImageFromArray(np.swapaxes(zeds,0,2))
    ref.SetSpacing((output_spacing,output_spacing,output_spacing))
    ref.SetOrigin(im.GetOrigin())
    ref.SetDirection(im.GetDirection())
    rs=sitk.ResampleImageFilter()
    rs.SetDefaultPixelValue(-1000)
    rs.SetInterpolator(sitk.sitkBSplineResampler)
    rs.SetReferenceImage(ref)
    out=rs.Execute(im)
    return out
def sobel_filter(im,threshold=-15):
    ar=sitk.GetArrayFromImage(im)
    ar15=(ar>-15).astype('float32')
    s15=abs(sobel(ar15,axis=0))+abs(sobel(ar15,axis=1))+abs(sobel(ar15,axis=2))
    s15=s15*5
    s15+=ar
    im15=match_array_to_im(s15,im)
    return im15
def read_itk_as_vtk_metareader(im):
    temp_file='temp.mhd'
    #im=sitk.ReadImage(filename)
    sitk.WriteImage(im,temp_file)
    v16=vtk.vtkMetaImageReader()
    v16.SetFileName(temp_file)
    v16.Update()
    os.unlink(temp_file)
    os.unlink(temp_file.replace('.mhd','.raw'))
    return v16
def vtk_surface_mapper(volume_reader,threshold,color_list,specular=0.3,
                       specular_power=20,opacity=0.5):
    colors=vtk.vtkNamedColors()
    colors.SetColor("Color", color_list)
    surfaceExtractor = vtk.vtkMarchingCubes()
    #surfaceExtractor.SetInputConnection(volume_reader.GetOutputPort())
    surfaceExtractor.SetInputData(volume_reader)
    surfaceExtractor.SetValue(0, threshold)

    surfaceStripper = vtk.vtkStripper()
    surfaceStripper.SetInputConnection(surfaceExtractor.GetOutputPort())
    surfaceMapper = vtk.vtkPolyDataMapper()
    surfaceMapper.SetInputConnection(surfaceStripper.GetOutputPort())
    #surfaceMapper.SetInputConnection(decimated.GetOutputPort())
    surfaceMapper.ScalarVisibilityOff()
    
##    decimate = vtk.vtkDecimatePro()  
##    decimate.SetInputData(surfaceMapper.GetOutputPort())
##    decimate.SetTargetReduction(0.5)
##    decimate.PreserveTopologyOn()
##    decimate.Update()
##    decimated = vtk.vtkPolyData()
##    decimated.ShallowCopy(decimate.GetOutput())

    surface = vtk.vtkActor()
    surface.SetMapper(surfaceMapper)
    surface.GetProperty().SetDiffuseColor(colors.GetColor3d("Color"))
    surface.GetProperty().SetSpecular(specular)
    surface.GetProperty().SetSpecularPower(specular_power)
    surface.GetProperty().SetOpacity(opacity)
    return surface

def vtk_isosurface_mapper(vtk_image,n_surfaces,max_alpha=0.5):
    iso_values=np.linspace(vtk_image.GetScalarRange()[0],vtk_image.GetScalarRange()[1],num=n_surfaces)
    alpha_values=np.linspace(0,max_alpha,num=n_surfaces)
    surfaces=[]
    for i in range(n_surfaces):
        surfaces.append(vtk_surface_mapper(vtk_image,iso_values[i],[200,100,100,100],100.0,0.0,alpha_values[i]))
    return surfaces
            
def read_itk_thru_numpy(im):
    #im=sitk.ReadImage(filename)
    vol=sitk.GetArrayFromImage(im)
    [h, w, z] = vol.shape
    vtk_vol = numpy_support.numpy_to_vtk(num_array=vol.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
    vtk_image = vtk.vtkImageData()
    vtk_image.GetPointData().SetScalars(vtk_vol)
    vtk_image.SetDimensions(im.GetSize())
    vtk_image.SetOrigin(im.GetOrigin())
    vtk_image.SetSpacing(im.GetSpacing())
    return vtk_image
def get_dice(gt,seg):
#    dice = np.sum(seg[gt==1])*2.0 / (np.sum(seg) + np.sum(gt))
    dice = np.sum(seg[gt==1]==1)*2.0 / (np.sum(seg[seg==1]==1) + np.sum(gt[gt==1]==1))
    return dice

def render_im_and_struct(qspect,struct_gt,struct_pred,ptmax=30,output_filename='label_accuracy.png'):
    v16=read_itk_as_vtk_metareader(qspect)
    qs_ar=sitk.GetArrayFromImage(qspect)
    gt_ar=sitk.GetArrayFromImage(struct_gt)
    pred_ar=sitk.GetArrayFromImage(struct_pred)
    dice_score=get_dice(gt_ar,pred_ar)
    struct_ar=np.zeros(gt_ar.shape)  #1 True positive, 2 false positive, 3 false negative
    struct_ar[np.logical_and((gt_ar>0),(pred_ar>0))]=1
    struct_ar[np.logical_and((gt_ar==0),(pred_ar>0))]=2
    struct_ar[np.logical_and((gt_ar>0),(pred_ar==0))]=3
    voxel_volume=np.prod(np.array(struct_gt.GetSpacing()))/1000. #volume in cc
    true_pos_vol=round((struct_ar==1).sum()*voxel_volume,1)
    false_pos_vol=round((struct_ar==2).sum()*voxel_volume,1)
    false_neg_vol=round((struct_ar==3).sum()*voxel_volume,1)
    vol_gt=round((gt_ar>0).sum()*voxel_volume,1)
    vol_pred=round((pred_ar>0).sum()*voxel_volume,1)
    mean_gt=round(qs_ar[gt_ar>0].mean(),2)
    mean_pred=round(qs_ar[pred_ar>0].mean(),2)
    surfaces=[]
    last_number=255
    colorlists=[[0,0,255,last_number],[0,255,0,last_number],[255,0,0,last_number],[76,0,153,last_number]]
    opacities=[0,0.15,0.4,0.4]
    for value in np.unique(struct_ar):
        if value>0:
            current_ar=(struct_ar==value).astype('int8')
            current_struct=match_array_to_im(current_ar,struct_gt)
            current_struct=read_itk_thru_numpy(current_struct)
            surfaces.append(vtk_surface_mapper(current_struct,1,
                                               colorlists[int(value)],
                                               opacity=opacities[int(value)]))
    #surface=vtk_surface_mapper(struct,1,[100,180,255,255],opacity=0.2)
    
##    im=sitk.ReadImage(file_name)
##    im=resample_cubic(im,1.5)
##    im=sobel_filter(im,-15)
##    v16=read_itk_as_vtk_metareader(im)
##    sitk.WriteImage(im,'temp.mha')
##    file_name='temp.mha'

    #colors.SetColor('BkgColor', [51, 77, 102, 255])
    colors.SetColor('BkgColor', [255, 255, 255, 255])

    # Create the renderer, the render window, and the interactor. The renderer
    # draws into the render window, the interactor enables mouse- and
    # keyboard-based interaction with the scene.
    ren = vtk.vtkRenderer()
    ren_win = vtk.vtkRenderWindow()
    ren_win.AddRenderer(ren)
    
    s='Dice score: '+str(round(dice_score,3))+'\n'
    s+='Volumes - True+: '+str(true_pos_vol)+' - False+: '+str(false_pos_vol)+' - False-: '+str(false_neg_vol)+'\n'
    s+='True Volume: '+str(vol_gt)+' - Predicted Volume: '+str(vol_pred)+'\n'
    s+='True Mean SUV: '+str(mean_gt)+' - Predicted Mean SUV: '+str(mean_pred)
    txt=vtkTextActor()
    print(s)
    txt.SetInput(s)
    txtprop = txt.GetTextProperty()
    txtprop.SetFontFamilyToArial()
##    txtprop.BoldOn()
    txtprop.SetFontSize(24)
##    txtprop.ShadowOn()
##    txtprop.SetShadowOffset(4, 4)
    txtprop.SetColor(colors.GetColor3d('Black'))
    txt.SetDisplayPosition(20, 30)    

    
    #vtk_image = vtk_win_im.GetOutput()
    
##    iren = vtk.vtkRenderWindowInteractor()
##    iren.SetRenderWindow(ren_win)

    # The following reader is used to read a series of 2D slices (images)
    # that compose the volume. The slice dimensions are set, and the
    # pixel spacing. The data Endianness must also be specified. The reader
    # uses the FilePrefix in combination with the slice number to construct
    # filenames using the format FilePrefix.%d. (In this case the FilePrefix
    # is the root name of the file: quarter.)
##    reader = vtk.vtkMetaImageReader()
##    reader.SetFileName(file_name)

    # The volume will be displayed by ray-cast alpha compositing.
    # A ray-cast mapper is needed to do the ray-casting.
    volume_mapper = vtk.vtkFixedPointVolumeRayCastMapper()
    #volume_mapper.SetInputConnection(reader.GetOutputPort())
    volume_mapper.SetInputConnection(v16.GetOutputPort())

    # The color transfer function maps voxel intensities to colors.
    # It is modality-specific, and often anatomy-specific as well.
    # The goal is to one color for flesh (between 500 and 1000)
    # and another color for bone (1150 and over).
    volume_color = vtk.vtkColorTransferFunction()
    volume_color.AddRGBPoint(0, 0.0, 0.0, 0.0)
    volume_color.AddRGBPoint(ptmax, 0.0, 0.0, 0.0)
    
    
##    volume_color.AddRGBPoint(ptmax/4., 0, 0, 204/255.)
##    volume_color.AddRGBPoint(ptmax/2., 0, 153/255., 76/255.)
##    volume_color.AddRGBPoint(3*ptmax/4, 204/255., 204/255., 0)  # Ivory
##    volume_color.AddRGBPoint(ptmax, 255/255., 0, 0)  # Ivory

    # The opacity transfer function is used to control the opacity
    # of different tissue types.
    volume_scalar_opacity = vtk.vtkPiecewiseFunction()
    volume_scalar_opacity.AddPoint(0, 0.00)
    volume_scalar_opacity.AddPoint(ptmax/10.,0.15)
    volume_scalar_opacity.AddPoint(ptmax,0.5)
##    volume_scalar_opacity.AddPoint(500, 0.10)
##    volume_scalar_opacity.AddPoint(1000, 0.25)
##    volume_scalar_opacity.AddPoint(1150, 0.85)

    # The gradient opacity function is used to decrease the opacity
    # in the 'flat' regions of the volume while maintaining the opacity
    # at the boundaries between tissue types.  The gradient is measured
    # as the amount by which the intensity changes over unit distance.
    # For most medical data, the unit distance is 1mm.
    volume_gradient_opacity = vtk.vtkPiecewiseFunction()
    volume_gradient_opacity.AddPoint(0, 0.0)
##    volume_gradient_opacity.AddPoint(90, 0.5)
    volume_gradient_opacity.AddPoint(30, 0.5)
    volume_gradient_opacity.AddPoint(100, 1.0)

    # The VolumeProperty attaches the color and opacity functions to the
    # volume, and sets other volume properties.  The interpolation should
    # be set to linear to do a high-quality rendering.  The ShadeOn option
    # turns on directional lighting, which will usually enhance the
    # appearance of the volume and make it look more '3D'.  However,
    # the quality of the shading depends on how accurately the gradient
    # of the volume can be calculated, and for noisy data the gradient
    # estimation will be very poor.  The impact of the shading can be
    # decreased by increasing the Ambient coefficient while decreasing
    # the Diffuse and Specular coefficient.  To increase the impact
    # of shading, decrease the Ambient and increase the Diffuse and Specular.
    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetColor(volume_color)
    volume_property.SetScalarOpacity(volume_scalar_opacity)
    volume_property.SetGradientOpacity(volume_gradient_opacity)
    volume_property.SetInterpolationTypeToLinear()
    volume_property.ShadeOn()
    volume_property.SetAmbient(0.4)
    volume_property.SetDiffuse(0.6)
    volume_property.SetSpecular(0.2)

    # The vtkVolume is a vtkProp3D (like a vtkActor) and controls the position
    # and orientation of the volume in world coordinates.
    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)

    # Finally, add the volume to the renderer
    ren.AddViewProp(volume)

    for i in range(len(surfaces)):
        ren.AddActor(surfaces[i])

    ren.AddActor(txt)
    # Set up an initial view of the volume.  The focal point will be the
    # center of the volume, and the camera position will be 400mm to the
    # patient's left (which is our right).
    camera = ren.GetActiveCamera()
    c = volume.GetCenter()
    #camera.SetViewUp(0, 0, -1)
    camera.SetViewUp(0, 0, 1)
    camera.SetPosition(c[0] + 350. - 2* int(epoch) , c[1] - 800., c[2]+400)
    #camera.SetPosition(c[0] + 500. - 2* int(epoch) , c[1] - 1200., c[2]+400)
    camera.SetFocalPoint(c[0], c[1], c[2])
    camera.Zoom(0.4)
    #camera.Azimuth(30.0 + int(epoch)/5.)
    #camera.Elevation(10.0)

    # Set a background color for the renderer
    ren.SetBackground(colors.GetColor3d('BkgColor'))

    # Increase the size of the render window
    ren_win.SetSize(800,1600)
    ren_win.SetWindowName('MedicalDemo4')
    ren_win.OffScreenRenderingOn()

    vtk_win_im = vtk.vtkWindowToImageFilter()
    vtk_win_im.SetInput(ren_win)
    vtk_win_im.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(output_filename)
    writer.SetInputData(vtk_win_im.GetOutput())
    writer.Write()
    statistics=[dice_score,true_pos_vol,false_pos_vol,false_neg_vol,vol_gt,vol_pred,mean_gt,mean_pred]
    return statistics


##statistics=render_im_and_struct(im,ttb,pred_ttb,20,'test_vtk.png')




