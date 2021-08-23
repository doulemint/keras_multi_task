from tensorflow.keras.applications import *
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
import tensorflow as tf
import os

def create_model():
    train_input_shape=(224,224,3)
    based_model = ResNet50(weights='imagenet', include_top=False, input_shape=train_input_shape)
    for layer in based_model.layers:
        layer.trainable = True
    # Add layers at the end
    X = based_model.output
    X = Flatten()(X)

    X = Dense(512, kernel_initializer='he_uniform')(X)
    X = Dropout(0.5)(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Dense(125, kernel_initializer='he_uniform')(X)
    X = Dropout(0.5)(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    output1 = Dense(Artist_class_num, name='Artist_output',activation='softmax')(X)
    output2 = Dense(Style_class_num, activation='softmax',name='Style_output')(X)
    output3 = Dense(Objtype_class_num, activation='softmax',name='Objtype_output')(X)
    #model = Model(inputs=based_model.input, outputs=[output1,output2,output3])
    output4 = Dense(CreationDate_class_num, activation='sigmoid',name='CreationDate_output')(X)
    model = Model(inputs=based_model.input, outputs=[output1,output2,output3,output4])

    return model

def generate_classdict(label):
  counter = Counter(label)
  class_num=len(counter)
  class_list=list(counter.keys()) #?
  class_dict={}
  class_weight={}
  total = len(label)
  count=0
  for name,num in counter.items():
    class_dict[name]=count
    class_weight[count]=(total/(num*class_num))
    count+=1
  return class_num,class_list,class_dict,class_weight
  