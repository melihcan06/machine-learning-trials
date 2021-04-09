import os
import sys
import random

import numpy as np
import cv2
import matplotlib.pyplot as plt

from keras.models import load_model
import tensorflow as tf
import keras

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization
from keras.optimizers import RMSprop,SGD,Adam
from keras.losses import binary_crossentropy
import keras.backend as K
from keras.layers import Conv2DTranspose

class Veriler:  # aau-rainsnow icin
    def __init__(self, yol, boyut=(128, 128, 3)):
        self.yol = yol
        self.boyut = boyut
        self.label_yol = yol + "rgbMasks/"

    def plt_ciz(self, goruntuler, satir=1, sutun=None):
        if sutun == None:
            sutun = len(goruntuler)
        for i in range(len(goruntuler)):
            plt.subplot(satir, sutun, i + 1)
            plt.imshow(goruntuler[i])
            plt.gray()
            plt.axis('off')
        plt.show()

    def gnck(self, alinacak_goruntu_sayisi, arka=1, i=0):  # goruntuleri_numpye_cevir#aynıları arka arkaya 3 kere koy
        x=np.array(os.listdir(self.yol))
        y=np.array(os.listdir(self.label_yol))
        x.sort()
        y.sort()
        grtler = []
        lbler = []

        resim_adlari=[]

        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                if x[i] == y[j]:
                    resim_adlari.append(x[i])
                    continue
        m=0
        alinan = 0
        while True:
            if alinan == alinacak_goruntu_sayisi:
                break             

            img1 = cv2.imread(self.yol + resim_adlari[m])            
            img2 = cv2.imread(self.label_yol + resim_adlari[m])                                    
            
            for j in range(arka):
                grtler.append(cv2.resize(img1, (self.boyut[0], self.boyut[1])))
                lbler.append(cv2.resize(img2, (self.boyut[0], self.boyut[1])))
                
            alinan += 1
            m+=1

        grtler = np.array(grtler)
        lbler = np.array(lbler)

        return grtler, lbler

def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

def unet(num_classes, input_shape, lr_init, vgg_weight_path=None): #renkli model
    img_input = Input(input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)
    block_1_out = Activation('relu')(x)

    x = MaxPooling2D()(block_1_out)

    # Block 2
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
    x = BatchNormalization()(x)
    block_2_out = Activation('relu')(x)

    x = MaxPooling2D()(block_2_out)

    # Block 3
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)
    x = BatchNormalization()(x)
    block_3_out = Activation('relu')(x)

    x = MaxPooling2D()(block_3_out)

    # Block 4
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x)
    x = BatchNormalization()(x)
    block_4_out = Activation('relu')(x)

    x = MaxPooling2D()(block_4_out)

    # Block 5
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block5_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block5_conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    for_pretrained_weight = MaxPooling2D()(x)

    # Load pretrained weights.
    if vgg_weight_path is not None:
        vgg16 = Model(img_input, for_pretrained_weight)
        vgg16.load_weights(vgg_weight_path, by_name=True)

    # UP 1
    x = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x, block_4_out])
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # UP 2
    x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x, block_3_out])
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # UP 3
    x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x, block_2_out])
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # UP 4
    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x, block_1_out])
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # last conv
    x = Conv2D(num_classes, (3, 3), activation='sigmoid', padding='same')(x)#activation='softmax'
    
    
    model = Model(img_input, x)
    model.compile(optimizer=Adam(lr=lr_init),
                  loss='categorical_crossentropy',#'mean_squared_error',#'categorical_crossentropy',
                  metrics=['accuracy'])#metrics=[dice_coef]
    return model
        
boyut=(128,128,3)
yol="dataset/Hjorringvej-3/"#"dataset/cityscapes_data/"#"dataset/hucre/"#"dataset/kitti/"#"dataset/insan2/"#
sinif_sayisi=3
alinacak_goruntu_sayisi=25#max = 2975 train verisi o kadar        

model=unet(sinif_sayisi,boyut,0.003)
model.summary()

veriler=Veriler(yol,boyut) 
x_train,y_train=veriler.gnck(500,arka=3)  

model.fit(x_train,y_train,batch_size=8,epochs=100,validation_split=0.2)
model.save("unetX")
