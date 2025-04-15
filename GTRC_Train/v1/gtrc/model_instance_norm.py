import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate #as concatenate
from tensorflow.keras.layers import Input, Convolution3D, MaxPooling3D, UpSampling3D
from tensorflow.keras.layers import Reshape, Activation, Dropout, ReLU, LeakyReLU
from tensorflow.keras.layers import BatchNormalization, Conv3D, AveragePooling3D, ZeroPadding3D
from tensorflow_addons.layers import InstanceNormalization

#Model design adapted from Shanghai Aitrox winning submission to MICCAI Flare 21 challenge
#similar to 3D unet with additional residual connections and anisotropic convolutions in decoder branch
#https://github.com/Shanghai-Aitrox-Technology/EfficientSegmentation
#written into tensorflow and pyramid pooling which didnt seem to improve training accuracy (at least for this task)


def ConvINReLU(input_layer, out_channels, kernel_size=1, stride=1, p=0.2, negative_slope=0.):
    #layer sequence in torch nn.Conv3d, nn.InstanceNorm3d, nn.Dropout3d, nn.ReLU
    #change negative_slope to 0.3 for default LeakyReLu behaviour
    x = Conv3D(out_channels,kernel_size=kernel_size, strides=stride,padding='same')(input_layer)
    x = InstanceNormalization()(x)
    x = Dropout(p)(x)
    x = ReLU(negative_slope=negative_slope)(x)
    return x

def ConvIN(input_layer, out_channels, kernel_size=1, stride=1):    
    #layer sequence in torch nn.Conv3d, nn.InstanceNorm3d
    x = Conv3D(out_channels,kernel_size=kernel_size, strides=stride,padding='same')(input_layer)
    x = InstanceNormalization()(x)
    return x

def ResFourLayerConvBlock(input_layer, inter_channel, out_channel, p=0.2, stride=1, negative_slope=0.):
    #residual_unit_1
    conv1 = ConvINReLU(input_layer,inter_channel, 3, stride=stride, p=p, negative_slope=negative_slope)
    conv1 = ConvIN(conv1,inter_channel, 3, stride=1)
    res1 = ConvIN(input_layer,inter_channel, 1, stride=stride)
    conv1 = Concatenate(axis=-1)([conv1,res1])
    conv1 = ReLU(negative_slope=negative_slope)(conv1)
    #residual_unit_2
    conv2 = ConvINReLU(conv1,inter_channel, 3, stride=1, p=p, negative_slope=negative_slope)
    conv2 = ConvIN(conv2,out_channel, 3, stride=1)
    #res2 = ConvIN(input_layer,inter_channel, 1, stride=1) #residual unit 2 is just conv1 getting passed around
    conv2 = Concatenate(axis=-1)([conv2,conv1])
    conv2 = ReLU(negative_slope=negative_slope)(conv2)
    return conv2

    
def AnisotropicConvBlock(input_layer, out_channel, p=0.2, stride=1,negative_slope=0):
    #residual unit:
    #   ConvINReLU3D(in_channel, out_channel, kernel_size=(3, 3, 1), stride=stride, padding=(1, 1, 0), p=p)
    #   ConvIN3D(out_channel, out_channel, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1)))
    #shortcut unit:
    #   ConvIN3D(in_channel, out_channel, kernel_size=1, stride=stride, padding=0)
    #relu:
    #   ReLU(inplace=True)    
    conv1 = ConvINReLU(input_layer,out_channel, (3,3,1), stride=stride, p=p, negative_slope=negative_slope)
    conv1 = ConvIN(conv1,out_channel, (1,1,3), stride=1)
    res1 = ConvIN(input_layer,out_channel, 1, stride=1)
    conv1 = Concatenate(axis=-1)([conv1,res1])
    conv1 = ReLU(negative_slope=negative_slope)(conv1)
    return conv1

def BuildModel(input_size=(128,128,256,1),negative_slope=0.,filter_depths=[16, 32, 64, 128, 256],output_channels=2,p=0.2):
    data=Input(shape=input_size)
    x0=ResFourLayerConvBlock(data, filter_depths[0], filter_depths[0], p=0.2, stride=1, negative_slope=negative_slope)
    x1=ResFourLayerConvBlock(x0, filter_depths[1], filter_depths[1], p=0.2, stride=2, negative_slope=negative_slope)
    x2=ResFourLayerConvBlock(x1, filter_depths[2], filter_depths[2], p=0.2, stride=2, negative_slope=negative_slope)
    x3=ResFourLayerConvBlock(x2, filter_depths[3], filter_depths[3], p=0.2, stride=2, negative_slope=negative_slope)
    x4=ResFourLayerConvBlock(x3, filter_depths[4], filter_depths[4], p=0.2, stride=2, negative_slope=negative_slope)
    x4=ConvINReLU(x4, filter_depths[4], 1, stride=1, p=p, negative_slope=negative_slope)
    u4=ConvINReLU(x4, filter_depths[3], 1, stride=1, p=p, negative_slope=negative_slope) #squeeze filters before concat
    u3=UpSampling3D(2)(u4) #[24,24,24,128]
    u3=Concatenate()([u3,x3])
    u3=AnisotropicConvBlock(u3, filter_depths[3], p=0.2, stride=1) 
    u2=UpSampling3D(2)(u3) #[48,48,48,64]
    u2=Concatenate()([u2,x2])    
    u2=AnisotropicConvBlock(u2, filter_depths[2], p=0.2, stride=1) 
    u1=UpSampling3D(2)(u2) #[96,96,96,32]
    u1=Concatenate()([u1,x1])    
    u1=AnisotropicConvBlock(u1, filter_depths[1], p=0.2, stride=1) 
    u0=UpSampling3D(2)(u1) #[192,192,192,16]
    u0=Concatenate()([u0,x0])        
    u0=AnisotropicConvBlock(u0, filter_depths[0], p=0.2, stride=1) 
    logit = Conv3D(output_channels,kernel_size=1, strides=1,padding='same')(u0)
    logit = Activation('softmax')(logit) 
    model = Model(data, logit)
    return model
