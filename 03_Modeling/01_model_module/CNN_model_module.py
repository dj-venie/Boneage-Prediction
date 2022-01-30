# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 10:36:50 2020

@author: Administrator
"""

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *

from tensorflow.keras import layers
from tensorflow.keras import models


# 모델
from tensorflow.keras.metrics import mean_absolute_error

def mae_in_months(x_p, y_p):
    '''function to return mae in months'''
    return mean_absolute_error((ba_std*x_p + ba_mean), (ba_std*y_p + ba_mean))  

# xception 기반모델
def xception(img_size = 256):
    model_1 = tf.keras.applications.xception.Xception(input_shape = (img_size, img_size, 3),
                                               include_top = False,
                                               weights = 'imagenet')
    model_1.trainable = True
    model = Sequential()
    model.add(model_1)
    model.add(GlobalMaxPooling2D())
    model.add(Flatten())
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(1, activation = 'linear'))
    
    return model

# vgg와 ba_net 합친모델
def vgg_ba(img_size = 224):
    model = Sequential()

    model.add(Conv2D(8,kernel_size =(3,3),
                    input_shape = (img_size,img_size,3),activation='relu'))
    model.add(Conv2D(8,(3,3),activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.1))
    
    model.add(Conv2D(16,(3,3),activation = 'relu'))
    model.add(Conv2D(16,(3,3),activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.25))
    
    model.add(Conv2D(32,(3,3),activation = 'relu'))
    model.add(Conv2D(32,(3,3),activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(64,(3,3),activation = 'relu'))
    model.add(Conv2D(64,(3,3),activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    
    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(1))
    
    return model

# 성별을 추가 하기위해 만든 모델
def gender(act='relu'):
    g_input = Input(shape = (1,))
    g_output = Dense(64,activation = act)(g_input)
    
    g_model = Model(inputs = g_input,outputs = g_output)
    return g_model

# xception+gender 2개의 input이라 2를 붙임
def xception2(img_size = 256):
    model_1 = tf.keras.applications.xception.Xception(input_shape = (img_size, img_size, 3),
                                               include_top = False,
                                               weights = 'imagenet')
    model_1.trainable = True
    model_2 = Sequential()
    model_2.add(model_1)
    model_2.add(GlobalMaxPooling2D())
    model_2.add(Flatten())
    model_2.add(Dense(2048, activation = 'relu'))
    
    g_model = gender()
    
    con = concatenate([g_model.output, model_2.output])
    dense1 = Dense(64,activation='relu')(con)
    batch = BatchNormalization()(dense1)
    drop = Dropout(0.5)(batch)
    dense2 = Dense(10,activation='softmax')(drop)
    model_out = Dense(1,activation = "linear")(dense2)
    
    model = Model([g_model.input,model_2.input],model_out)
    return model

# vgg+gender vgg(16,19) 가능
def vgg2(n=16,img_size = 224):
    
    g_model = gender()
    
    if n==19:
        model_1 = tf.keras.applications.vgg19.VGG19(input_shape = (img_size, img_size, 3),
                                                   include_top = False,
                                                   weights = 'imagenet')
    elif n==16:
        model_1 = tf.keras.applications.vgg16.VGG16(input_shape = (img_size, img_size, 3),
                                                   include_top = False,
                                                   weights = 'imagenet')
    else:
        return print("n must be 16 or 19")
    
    model_1.trainable = True
    model_2 = Sequential()
    model_2.add(model_1)
    model_2.add(GlobalMaxPooling2D())
    model_2.add(Flatten())
    model_2.add(Dense(2048, activation = 'relu'))
    
    con = concatenate([g_model.output, model_2.output])
    
    dense1 = Dense(1024,activation='relu')(con)
    dense2 = Dense(512,activation='relu')(dense1)
    model_out = Dense(1,activation = "linear")(dense2)
    
    model = Model([g_model.input,model_2.input],model_out)
    return model

# resnet+gender resnet(50,101,151) 가능
def resnet2(n=50,img_size = 224):
    
    g_model = gender()
    
    if n==50:
        model_1 = tf.keras.applications.resnet.ResNet50(input_shape = (img_size, img_size, 3),
                                               include_top = False,
                                               weights = 'imagenet')
    elif n==101:
        model_1 = tf.keras.applications.resnet.ResNet101(input_shape = (img_size, img_size, 3),
                                               include_top = False,
                                               weights = 'imagenet')
    elif n==151:
        model_1 = tf.keras.applications.resnet.ResNet151(input_shape = (img_size, img_size, 3),
                                               include_top = False,
                                               weights = 'imagenet')
    else:
        return print("n must be 50 or 101 or 151")
    model_1.trainable = True
    model_2 = Sequential()
    model_2.add(model_1)
    model_2.add(GlobalMaxPooling2D())
    model_2.add(Flatten())
    model_2.add(Dense(2048, activation = 'relu'))
    
    con = concatenate([g_model.output, model_2.output])
    
    dense1 = Dense(1024,activation='relu')(con)
    dense2 = Dense(512,activation='relu')(dense1)
    model_out = Dense(1,activation = "linear")(dense2)
    
    model = Model([g_model.input,model_2.input],model_out)
    return model

# tjnet 기반으로 만든모델
def tjnet2(img_size = 251):
    
    
    i_input = Input(shape = (img_size,img_size,3))
    #block 1
    conv1 = ReLU()(Conv2D(32,3)(i_input))
    
    conv2 = ReLU()(Conv2D(32,3)(conv1))
    conv3 = ReLU()(Conv2D(64,3,padding='same')(conv2))
    pool1 = MaxPool2D(pool_size=(2,2))(conv3)
    #block 2
    conv4 = ReLU()(Conv2D(60,1)(pool1))
    conv5 = ReLU()(Conv2D(192,3)(conv4))
    pool2 = MaxPool2D(pool_size=(2,2))(conv5)
    #block 3
    conv6 = ReLU()(Conv2D(512,3)(pool2))
    pool3 = MaxPool2D(pool_size=(2,2))(conv6)
    #block 4
    conv7 = ReLU()(Conv2D(1024,3,padding='same')(pool3))
    pool4 = MaxPool2D(pool_size=(2,2))(conv7)
    #block 5
    conv8 = ReLU()(Conv2D(2048,3)(pool4))
    conv9 = ReLU()(Conv2D(2048,5)(conv8))
    pool5 = AveragePooling2D(pool_size=(8,8))(conv9)
    i_output = Flatten()(pool5)

    
    i_model = Model(inputs = i_input, outputs = i_output)
    
    g_model = gender()
    con = concatenate([g_model.output, i_model.output])
    dense1 = ReLU()(Dense(1024)(con))

    dense2 = ReLU()(Dense(512)(dense1))

    model_out = Dense(1,activation = "linear")(dense2)
    
    model = Model([g_model.input,i_input],model_out)
    return model


def tjnetse2(img_size = 251):
    
    i_input = Input(shape = (img_size,img_size,3))
    #block 1
    conv1 = ReLU()(Conv2D(32,3)(i_input))
    conv2 = ReLU()(Conv2D(32,3)(conv1))
    conv3 = ReLU()(Conv2D(64,3,padding='same')(conv2))
    x = Conv2D(64,1)(conv3)
    x = BatchNormalization()(x)

    se = GlobalAveragePooling2D()(x)
    se = Dense(4,activation="relu")(se)
    se = Dense(64,activation="relu")(se)
    se = Reshape([1,1,64])(se)
    x = Multiply()([conv3,se])

    short = Conv2D(64,(1,1))(conv2)
    short = BatchNormalization()(short)

    x = ReLU()(add([x,short]))

    pool1 = MaxPool2D(pool_size=(2,2))(x)
    #block 2
    conv4 = ReLU()(Conv2D(60,1)(pool1))
    conv5 = ReLU()(Conv2D(192,3)(conv4))
    pool2 = MaxPool2D(pool_size=(2,2))(conv5)
    #block 3
    conv6 = ReLU()(Conv2D(512,3)(pool2))
    pool3 = MaxPool2D(pool_size=(2,2))(conv6)
    #block 4
    conv7 = ReLU()(Conv2D(1024,3,padding='same')(pool3))
    pool4 = MaxPool2D(pool_size=(2,2))(conv7)
    #block 5
    conv8 = ReLU()(Conv2D(2048,3)(pool4))
    conv9 = ReLU()(Conv2D(2048,5)(conv8))
    pool5 = AveragePooling2D(pool_size=(8,8))(conv9)
    i_output = Flatten()(pool5)
       
    i_model = Model(inputs = i_input, outputs = i_output)
    
    g_model = gender()
    con = concatenate([g_model.output, i_model.output])
    dense1 = ReLU()(Dense(1024)(con))

    dense2 = ReLU()(Dense(512)(dense1))

    model_out = Dense(1,activation = "linear")(dense2)
    
    model = Model([g_model.input,i_input],model_out)
    return model


# mobilenet 기반모델
def mobile2(img_size = 224):
    mob_model = tf.keras.applications.MobileNet(input_shape = (img_size, img_size, 3),
                                               include_top = False,
                                               weights = 'imagenet')
    mob_model.trainable = True
    model_2 = Sequential()
    model_2.add(mob_model)
    model_2.add(GlobalMaxPooling2D())
    model_2.add(Flatten())
    model_2.add(Dense(2048, activation = 'relu'))
    
    g_model = gender()
    
    con = concatenate([g_model.output, model_2.output])
    dense1 = ReLU()(Dense(1024)(con))
    dense2 = ReLU()(Dense(512)(dense1))
    
    model_out = Dense(1,activation = "linear")(dense2)
    model = Model([g_model.input,model_2.input],model_out)
    
    return model


# tjnet(roi)+mobile(total)+gender 3가지 input
def tjnet3(bimg_size = 224,img_size = 251):
    #원본이미지 모델 (mobilnet)
    

    mob_model = tf.keras.applications.MobileNet(input_shape = (bimg_size, bimg_size, 3),
                                                   include_top = False,
                                                   weights = 'imagenet')
    mob_model.trainable = True
    model_2 = Sequential()
    model_2.add(mob_model)
    model_2.add(GlobalAveragePooling2D())
    model_2.add(Dense(2048, activation = 'relu'))

    # roi이미지 모델 (tj-net)

    i_input = Input(shape = (img_size,img_size,3))
    
    #block 1
    conv1 = ReLU()(Conv2D(32,3)(i_input))
    conv2 = ReLU()(Conv2D(32,3)(conv1))
    conv3 = ReLU()(Conv2D(64,3,padding='same')(conv2))
    x = Conv2D(64,1)(conv3)
    x = BatchNormalization()(x)

    se = GlobalAveragePooling2D()(x)
    se = Dense(4,activation="relu")(se)
    se = Dense(64,activation="relu")(se)
    se = Reshape([1,1,64])(se)
    x = Multiply()([conv3,se])

    short = Conv2D(64,(1,1))(conv2)
    short = BatchNormalization()(short)

    x = ReLU()(add([x,short]))

    pool1 = MaxPool2D(pool_size=(2,2))(x)
    #block 2
    conv4 = ReLU()(Conv2D(60,1)(pool1))
    conv5 = ReLU()(Conv2D(192,3)(conv4))
    pool2 = MaxPool2D(pool_size=(2,2))(conv5)
    #block 3
    conv6 = ReLU()(Conv2D(512,3)(pool2))
    pool3 = MaxPool2D(pool_size=(2,2))(conv6)
    #block 4
    conv7 = ReLU()(Conv2D(1024,3,padding='same')(pool3))
    pool4 = MaxPool2D(pool_size=(2,2))(conv7)
    #block 5
    conv8 = ReLU()(Conv2D(2048,3)(pool4))
    conv9 = ReLU()(Conv2D(2048,5)(conv8))
    pool5 = AveragePooling2D(pool_size=(8,8))(conv9)
    i_output = Flatten()(pool5)

    i_model = Model(inputs = i_input, outputs = i_output)

    # gender
    g_model = gender()

    #concat및 마무리
    con1 = concatenate([model_2.output, i_model.output])
    dense1 = ReLU()(Dense(2048)(con1))
    con2 = concatenate([dense1,g_model.output])
    dense2 = ReLU()(Dense(1024)(con2))
    dense3 = ReLU()(Dense(512)(dense2))
    model_out = Dense(1,activation = "linear")(dense3)

    model = Model([model_2.input,i_model.input,g_model.input],model_out)
    return model
    
### 동준
### 성준
    
def vgg_model(img_size=251):
    model = models.Sequential()
    model.add(Conv2D(input_shape=(img_size,img_size,3),filters=32,kernel_size=(3,3),padding="same", activation="elu"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32,kernel_size=(1,1),padding="same", activation="elu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

    model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="elu"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=(1,1), padding="same", activation="elu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="elu"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=(1,1), padding="same", activation="elu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="elu"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=(1,1), padding="same", activation="elu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="elu"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=256, kernel_size=(1,1), padding="same", activation="elu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

    model.add(Conv2D(filters=384, kernel_size=(3,3), padding="same", activation="elu"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=384, kernel_size=(1,1), padding="same", activation="elu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(Flatten())
    
    # gender
    g_model = gender(act='selu')

    con = concatenate([g_model.output,model.output])
    dense1 = Dense(2048,activation='relu')(con)
    dense1 = Dense(2048,activation='relu')(dense1)
    dense2 = Dense(256,activation='relu')(dense1)
    model_out = Dense(1,activation = "linear")(dense2)

    model = Model([g_model.input,model.input],model_out)
    return model




def se_resnet(img_size=251,dense_activation='sigmoid'):
    i_input = Input(shape=(img_size, img_size, 3))

    def conv1_layer(x):    
        x = ZeroPadding2D(padding=(3, 3))(x)
        x = Conv2D(64, (7, 7), strides=(2, 2))(x)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        x = ZeroPadding2D(padding=(1,1))(x)

        return x   



    def conv2_layer(x):         
        x = MaxPooling2D((3, 3), 2)(x)     

        shortcut = x

        for i in range(3):
            if (i == 0):
                x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
                x = BatchNormalization()(x)
                x = Activation('elu')(x)

                x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation('elu')(x)

                x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
                shortcut = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(shortcut)            
                x = BatchNormalization()(x)

                se = GlobalAveragePooling2D()(x)
                se = Dense(16, activation='elu' )(se)
                se = Dense(256, activation=dense_activation)(se)
                se = Reshape([1, 1,  256])(se)

                x = Multiply()([x,se])
                shortcut = BatchNormalization()(shortcut)

                x = Add()([x,shortcut])
                x = Activation('elu')(x)
                x = BatchNormalization()(x)
                shortcut = x

            else:
                x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
                x = BatchNormalization()(x)
                x = Activation('elu')(x)

                x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation('elu')(x)

                x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
                x = BatchNormalization()(x) 

                se = GlobalAveragePooling2D()(x)
                se = Dense(16, activation='elu' )(se)
                se = Dense(256, activation=dense_activation)(se)
                se = Reshape([1, 1,  256])(se)

                x = Multiply()([x,se])
                shortcut = BatchNormalization()(shortcut)

                x = Add()([se,shortcut]) 
                x = Activation('elu')(x)  
                x = BatchNormalization()(x)
                shortcut = x        

        return x



    def conv3_layer(x):        
        shortcut = x    

        for i in range(4):     
            if(i == 0):            
                x = Conv2D(128, (1, 1), strides=(2, 2), padding='valid')(x)
                x = BatchNormalization()(x)
                x = Activation('elu')(x)        

                x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation('elu')(x)  

                x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
                shortcut = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(shortcut)
                x = BatchNormalization()(x)

                se = GlobalAveragePooling2D()(x)
                se = Dense(16, activation='elu' )(se)
                se = Dense(512, activation=dense_activation)(se)
                se = Reshape([1, 1,  512])(se)

                x = Multiply()([x,se])
                shortcut = BatchNormalization()(shortcut)            

                x = Add()([se,shortcut])  
                x = Activation('elu')(x)    
                x = BatchNormalization()(x)
                shortcut = x              

            else:
                x = Conv2D(128, (1, 1), strides=(1, 1), padding='valid')(x)
                x = BatchNormalization()(x)
                x = Activation('elu')(x)

                x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation('elu')(x)

                x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
                x = BatchNormalization()(x)            

                se = GlobalAveragePooling2D()(x)
                se = Dense(16, activation='elu' )(se)
                se = Dense(512, activation=dense_activation)(se)
                se = Reshape([1, 1,  512])(se)

                x = Multiply()([x,se])
                shortcut = BatchNormalization()(shortcut)

                x = Add()([se,shortcut])    
                x = Activation('elu')(x)
                x = BatchNormalization()(x)
                shortcut = x      

        return x



    def conv4_layer(x):
        shortcut = x        

        for i in range(23):     
            if(i == 0):            
                x = Conv2D(256, (1, 1), strides=(2, 2), padding='valid')(x)
                x = BatchNormalization()(x)
                x = Activation('elu')(x)        

                x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation('elu')(x)  

                x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x)
                shortcut = Conv2D(1024, (1, 1), strides=(2, 2), padding='valid')(shortcut)
                x = BatchNormalization()(x)

                se = GlobalAveragePooling2D()(x)
                se = Dense(16, activation='elu' )(se)
                se = Dense(1024, activation=dense_activation)(se)
                se = Reshape([1, 1,  1024])(se)

                x = Multiply()([x,se])
                shortcut = BatchNormalization()(shortcut)

                x = Add()([se,shortcut])
                x = Activation('elu')(x)
                x = BatchNormalization()(x)
                shortcut = x               

            else:
                x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
                x = BatchNormalization()(x)
                x = Activation('elu')(x)

                x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation('elu')(x)

                x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x)
                x = BatchNormalization()(x)    

                se = GlobalAveragePooling2D()(x)
                se = Dense(16, activation='elu' )(se)
                se = Dense(1024, activation=dense_activation)(se)
                se = Reshape([1, 1,  1024])(se)

                x = Multiply()([x,se])
                shortcut = BatchNormalization()(shortcut)

                x = Add()([se,shortcut])               
                x = Activation('elu')(x)
                x = BatchNormalization()(x)
                shortcut = x      

        return x



    def conv5_layer(x):
        shortcut = x    

        for i in range(3):     
            if(i == 0):            
                x = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(x)
                x = BatchNormalization()(x)
                x = Activation('elu')(x)        

                x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation('elu')(x)  

                x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')(x)
                shortcut = Conv2D(2048, (1, 1), strides=(2, 2), padding='valid')(shortcut)
                x = BatchNormalization()(x)

                se = GlobalAveragePooling2D()(x)
                se = Dense(16, activation='elu' )(se)
                se = Dense(2048, activation=dense_activation)(se)
                se = Reshape([1, 1,  2048])(se)

                x = Multiply()([x,se])
                shortcut = BatchNormalization()(shortcut)            

                x = Add()([se, shortcut])  
                x = Activation('elu')(x)      
                x = BatchNormalization()(x)
                shortcut = x               

            else:
                x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
                x = BatchNormalization()(x)
                x = Activation('elu')(x)

                x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation('elu')(x)

                x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')(x)
                x = BatchNormalization()(x)           

                se = GlobalAveragePooling2D()(x)
                se = Dense(16, activation='elu' )(se)
                se = Dense(2048, activation=dense_activation)(se)
                se = Reshape([1, 1,  2048])(se)

                x = Multiply()([x,se])
                shortcut = BatchNormalization()(shortcut)            

                x = Add()([se, shortcut]) 
                x = Activation('elu')(x)       
                x = BatchNormalization()(x)
                shortcut = x                  

        return x
 
 
    x = conv1_layer(i_input)
    x = conv2_layer(x)
    x = conv3_layer(x)
    x = conv4_layer(x)
    x = conv5_layer(x)
    x = GlobalAveragePooling2D()(x)
    i_output = Flatten()(x) 
    i_model = Model(inputs = i_input, outputs = i_output)    

    g_model = gender(act='selu')

    con = concatenate([g_model.output, i_model.output])
    dense1 = Dense(2048,activation='relu')(con)

    
    dense2 = Dense(256,activation='relu')(dense1)

    
    model_out = Dense(1, activation='linear')(dense2)


    model = Model([g_model.input,i_input],model_out)
    return model


def vgg16x2_model(img_size=251):
    
    g_model = gender(act='selu')
    model = models.Sequential()
    model.add(Conv2D(input_shape=(img_size,img_size,3),filters=32,kernel_size=(3,3),padding="same", activation="elu"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32,kernel_size=(1,1),padding="same", activation="elu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

    model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="elu"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=(1,1), padding="same", activation="elu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="elu"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=(1,1), padding="same", activation="elu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="elu"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=(1,1), padding="same", activation="elu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="elu"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=256, kernel_size=(1,1), padding="same", activation="elu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

    model.add(Conv2D(filters=384, kernel_size=(3,3), padding="same", activation="elu"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=384, kernel_size=(1,1), padding="same", activation="elu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(Flatten())

    model1 = models.Sequential()
    model1.add(Conv2D(input_shape=(img_size,img_size,3),filters=32,kernel_size=(3,3),padding="same", activation="elu"))
    model1.add(BatchNormalization())
    model1.add(Conv2D(filters=32,kernel_size=(1,1),padding="same", activation="elu"))
    model1.add(BatchNormalization())
    model1.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

    model1.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="elu"))
    model1.add(BatchNormalization())
    model1.add(Conv2D(filters=64, kernel_size=(1,1), padding="same", activation="elu"))
    model1.add(BatchNormalization())
    model1.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

    model1.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="elu"))
    model1.add(BatchNormalization())
    model1.add(Conv2D(filters=128, kernel_size=(1,1), padding="same", activation="elu"))
    model1.add(BatchNormalization())
    model1.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

    model1.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="elu"))
    model1.add(BatchNormalization())
    model1.add(Conv2D(filters=128, kernel_size=(1,1), padding="same", activation="elu"))
    model1.add(BatchNormalization())
    model1.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

    model1.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="elu"))
    model1.add(BatchNormalization())
    model1.add(Conv2D(filters=256, kernel_size=(1,1), padding="same", activation="elu"))
    model1.add(BatchNormalization())
    model1.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

    model1.add(Conv2D(filters=384, kernel_size=(3,3), padding="same", activation="elu"))
    model1.add(BatchNormalization())
    model1.add(Conv2D(filters=384, kernel_size=(1,1), padding="same", activation="elu"))
    model1.add(BatchNormalization())
    model1.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model1.add(Flatten())


    con = concatenate([g_model.output,model.output,model1.output])
    dense1 = Dense(2048,activation='relu')(con)

    dense2 = Dense(256,activation='relu')(dense1)

    model_out = Dense(1,activation = "linear")(dense2)

    model = Model([g_model.input,model.input,model1.input],model_out)
    
    return model
