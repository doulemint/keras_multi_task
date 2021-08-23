from utils import create_model
import pandas as pd
#drop unknow artist
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
from generator import CustomDataGen
from utils import create_model,generate_classdict

mpl.rcParams['figure.figsize'] = (22, 20)
pf=pd.read_csv('Preliminary_Training_Data_MoMA.csv')
indexName=pf[pf['Artist']=='Unknown photographer'].index
pf.drop(indexName,inplace=True)
grouped = pf.groupby(['Artist']).size().reset_index(name='counts')
p=grouped.sort_values('counts', ascending=False).head(100)
top50=p['Artist'].tolist()
dataset=pd.DataFrame()
for name,group in pf.groupby(['Artist']):
  if name in top50:
    dataset=pd.concat([dataset,group],axis=0)
dataset=dataset.reset_index()

import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit

X=np.array(dataset['imagefile'].tolist())
y=np.array(dataset['Style'].tolist())
Style_class_num,Style_class_list,Style_class_dict,Style_class_weight=generate_classdict(y)
y=np.array(dataset['Object Type'].tolist()) 
Objtype_class_num,Objtype_class_list,Objtype_class_dict,Objtype_class_weight=generate_classdict(y)
y=np.array(dataset['Creation Date'].tolist()) 
CreationDate_class_num,CreationDate_class_list,CreationDate_class_dict,CreationDate_class_weight=generate_classdict(y)
y1=np.array(dataset['Artist'].tolist()) 
Artist_class_num,Artist_class_list,Artist_class_dict,Artist_class_weight=generate_classdict(y)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
print(sss.get_n_splits(X, y1))
train_frame=pd.DataFrame()
test_frame=pd.DataFrame()
for train_index, test_index in sss.split(X, y1):
  train_frame=dataset.loc[train_index]
  test_frame=dataset.loc[test_index]

train_frame=train_frame.reset_index()
test_frame=test_frame.reset_index()

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np
path='./images/'
train_input_shape = (224,224)
batch_size=64
imgs_size=(64,224,224,3)


Artist_size=(batch_size,Artist_class_num)
Style_size=(batch_size,Style_class_num)
Objtype_size=(batch_size,Objtype_class_num)
CreationDate_size=(batch_size,CreationDate_class_num)

label_map={
  "Artist_class_num":Artist_class_num,
  "Style_class_num":Style_class_num,
  "Objtype_class_num":Objtype_class_num,
  "CreationDate_class_num":CreationDate_class_num,
  "Artist_class_dict":Artist_class_dict,
  "Style_class_dict":Style_class_dict,
  "Objtype_class_dict":Objtype_class_dict,
  "CreationDate_class_dict":CreationDate_class_dict
}

train_generator = tf.data.Dataset.from_generator(
     CustomDataGen(train_frame,batch_size,label_map,is_train=True,path=path),
     (tf.float64, {'Artist_output':tf.float32,'Style_output':tf.float32,'Objtype_output':tf.float32,'CreationDate_output':tf.float32}),
     (imgs_size, {'Artist_output':Artist_size,'Style_output':Style_size,'Objtype_output':Objtype_size,'CreationDate_output':CreationDate_size}))
valid_generator = tf.data.Dataset.from_generator(
     CustomDataGen(test_frame,batch_size,label_map,is_train=False,path=path),
     (tf.float64, {'Artist_output':tf.float32,'Style_output':tf.float32,'Objtype_output':tf.float32,'CreationDate_output':tf.float32}),
     (imgs_size, {'Artist_output':Artist_size,'Style_output':Style_size,'Objtype_output':Objtype_size,'CreationDate_output':CreationDate_size}))
#tf.TensorShape
#Load pre-train model
model = create_model()
optimizer = Adam(learning_rate=1e-4)
model.compile(loss={'Artist_output': 'categorical_crossentropy', 'Style_output': 'categorical_crossentropy', 'Objtype_output': 'categorical_crossentropy','CreationDate_output':'mean_squared_error'},
              optimizer=optimizer,
              loss_weights={'Artist_output':1,'Style_output':0.3,'Objtype_output':0.3,'CreationDate_output':0.3},
              metrics={'Artist_output':'accuracy','Style_output':'accuracy','Objtype_output':'accuracy','CreationDate_output':'accuracy'})
n_epoch=10
import os
import tempfile

def add_regularization(model, regularizer=tf.keras.regularizers.l2(0.0001)):

    if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
      print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
      return model

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
              setattr(layer, attr, regularizer)

    # When we change the layers attributes, the change only happens in the model config file
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    model.save_weights(tmp_weights_path)

    # load the model from the config
    model = tf.keras.models.model_from_json(model_json)
    
    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)
    return model
regular_rate=1e-4
#model=add_regularization(model,tf.keras.regularizers.l2(regular_rate))
#optimizer = Adam(lr=1e-4)
#model.compile(loss={'Artist_output': 'categorical_crossentropy', 'Style_output': 'categorical_crossentropy', 'Objtype_output': 'categorical_crossentropy'},
#              optimizer=optimizer,
#              loss_weights={'Artist_output':1,'Style_output':0.3,'Objtype_output':0.3},
#              metrics={'Artist_output':'accuracy','Style_output':'accuracy','Objtype_output':'accuracy'})
n_epoch=10
history1=model.fit(train_generator,
                  validation_data = valid_generator,steps_per_epoch=len(train_frame)//batch_size,
                  epochs=n_epoch,validation_steps=len(test_frame)//batch_size,
                  shuffle=True,
                  verbose = 1,
                  use_multiprocessing=True,
                  #callbacks=[red],
                  workers=16,)
#print(history1.history)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
def plot_metrics(history):
  epoch=range(len(history['Artist_output_accuracy']))
  metrics =  ['loss', 'accuracy']
  fig,axes = plt.subplots(2,3,figsize=(15,10))
  outputs=['Artist_output','Style_output','Objtype_output','CreationDate_output']
  for n, output in enumerate(outputs):
    metric=metrics[0]
    name = output.replace("_"," ").capitalize()
    plt.subplot(2,4,n+1)
    plt.plot(epoch, history[output+'_'+metric], color=colors[0], label='Train')
    plt.plot(epoch, history['val_'+output+'_'+metric],
             color=colors[0], linestyle="--", label='Val')
    name = metric.replace("_"," ").capitalize()
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.suptitle(name)
    metric=metrics[1]
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,4,n+5)
    plt.plot(epoch,history[output+'_'+metric], color=colors[0], label='Train')
    plt.plot(epoch,history['val_'+output+'_'+metric],
             color=colors[0], linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.suptitle(name)

    plt.legend()
  plt.savefig('training_plot.png')
#plot_metrics(history1)
save_dir = 'saved_models'
model_name = 'resnet50_art100_multitask.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
model.save(filepath)
early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, 
                           mode='auto', restore_best_weights=True)
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='Artist_output_accuracy',
                             verbose=1,
                             save_best_only=True)
for layer in model.layers[:50]:
    layer.trainable = False                               
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, 
                              verbose=1, mode='auto')
optimizer = Adam(lr=1e-5)
model.compile(loss={'Artist_output': 'categorical_crossentropy', 'Style_output': 'categorical_crossentropy', 'Objtype_output': 'categorical_crossentropy','CreationDate_output':'mean_squared_error'},
              optimizer=optimizer,
              loss_weights={'Artist_output':1,'Style_output':0.3,'Objtype_output':0.3,'CreationDate_output':0.3},
              metrics={'Artist_output':'accuracy','Style_output':'accuracy','Objtype_output':'accuracy','CreationDate_output':'accuracy'})
#model.compile(loss={'Artist_output': 'categorical_crossentropy', 'Style_output': 'categorical_crossentropy', 'Objtype_output': 'categorical_crossentropy'},
#              optimizer=optimizer,
#              loss_weights={'Artist_output':1,'Style_output':0.3,'Objtype_output':0.3},
#              metrics={'Artist_output':'accuracy','Style_output':'accuracy','Objtype_output':'accuracy'})
n_epoch=5
history2=model.fit(train_generator,
                  validation_data = valid_generator,steps_per_epoch=len(train_frame)//batch_size,
                  epochs=n_epoch,validation_steps=len(test_frame)//batch_size,
                  shuffle=False,
                  verbose = 2,
                  use_multiprocessing=True,
                  callbacks=[reduce_lr,early_stop,checkpoint],
                  workers=16,)
history = {}
history['Artist_output_loss'] = history1.history['Artist_output_loss'] + history2.history['Artist_output_loss']
history['Style_output_loss'] = history1.history['Style_output_loss'] + history2.history['Style_output_loss']
history['Objtype_output_loss'] = history1.history['Objtype_output_loss'] + history2.history['Objtype_output_loss']
history['CreationDate_output_loss'] = history1.history['CreationDate_output_loss'] + history2.history['CreationDate_output_loss']
history['val_Artist_output_loss'] = history1.history['val_Artist_output_loss'] + history2.history['val_Artist_output_loss']
history['val_Style_output_loss'] = history1.history['val_Style_output_loss'] + history2.history['val_Style_output_loss']
history['val_Objtype_output_loss'] = history1.history['val_Objtype_output_loss'] + history2.history['val_Objtype_output_loss']
history['val_CreationDate_output_loss'] = history1.history['val_CreationDate_output_loss'] + history2.history['val_CreationDate_output_loss']
history['Artist_output_accuracy'] = history1.history['Artist_output_accuracy'] + history2.history['Artist_output_accuracy']
history['Style_output_accuracy'] = history1.history['Style_output_accuracy'] + history2.history['Style_output_accuracy']
history['Objtype_output_accuracy'] = history1.history['Objtype_output_accuracy'] + history2.history['Objtype_output_accuracy']
history['CreationDate_output_accuracy'] = history1.history['CreationDate_output_accuracy'] + history2.history['CreationDate_output_accuracy']
history['val_Artist_output_accuracy'] = history1.history['val_Artist_output_accuracy'] + history2.history['val_Artist_output_accuracy']
history['val_Style_output_accuracy'] = history1.history['val_Style_output_accuracy'] + history2.history['val_Style_output_accuracy']
history['val_Objtype_output_accuracy'] = history1.history['val_Objtype_output_accuracy'] + history2.history['val_Objtype_output_accuracy']
history['val_CreationDate_output_accuracy'] = history1.history['val_CreationDate_output_accuracy'] + history2.history['val_CreationDate_output_accuracy']
print(history)
plot_metrics(history)
#######confusion matrix
# Classification report and confusion matrix
#from sklearn.metrics import *
#import matplotlib.pyplot as plt
#import seaborn as sns
#import numpy as np
#from tensorflow.keras.models import load_model
# class_dict={}
# index=0
# for name,num in count.items():
#   class_dict[name]=index
#   index+=1
#tick_labels = list(Artist_class_dict.keys())
#model=load_model('resnet50_art.h5')
#_multitask
#train_input_shape = (224,224)
#n_class = len(tick_labels)
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#batch_size = 64
#train_input_shape = (224,224)
#datagen = ImageDataGenerator(rescale = 1./255.,
#                horizontal_flip=False,
#                vertical_flip=False,)
#valid_generator=datagen.flow_from_dataframe(dataframe=test_frame,directory="./images",
#        x_col="imagefile",y_col="Artist",class_mode="categorical",
#        target_size=train_input_shape,batch_size=batch_size,classes=tick_labels)
#STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size
def showClassficationReport_Generator(model, valid_generator, STEP_SIZE_VALID):
    # Loop on each generator batch and predict
    y_pred, y_true = [], []
    for i in range(STEP_SIZE_VALID):
        (X,y) = next(valid_generator)
        y_pred.append(model.predict(X)[0])
        y_true.append(y)
    
    # Create a flat list for y_true and y_pred
    y_pred = [subresult for result in y_pred for subresult in result]
    y_true = [subresult for result in y_true for subresult in result]
    
    # Update Truth vector based on argmax
    y_true = np.argmax(y_true, axis=1)
    y_true = np.asarray(y_true).ravel()
    
    # Update Prediction vector based on argmax
    y_pred = np.argmax(y_pred, axis=1)
    y_pred = np.asarray(y_pred).ravel()
    
    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(50,50))
    conf_matrix = confusion_matrix(y_true, y_pred, labels=np.arange(n_class))
    conf_matrix = conf_matrix/np.sum(conf_matrix, axis=1)
    sns.heatmap(conf_matrix, annot=True, fmt=".2f", square=True, cbar=False, 
                cmap=plt.cm.jet, xticklabels=tick_labels, yticklabels=tick_labels,
                ax=ax)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    
    fig, ax = plt.subplots(figsize=(50,50))
    A=conf_matrix
    suppressed=A[~np.eye(A.shape[0],dtype=bool)].reshape(A.shape[0],-1) 
    #conf_matrix = conf_matrix/np.sum(conf_matrix, axis=1)
    sns.heatmap(suppressed, annot=True, fmt=".2f", square=True, cbar=False, 
                cmap=plt.cm.jet, xticklabels=tick_labels, yticklabels=tick_labels,
                ax=ax)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title('Offdiagonal Confusion Matrix')
    plt.savefig('offdiagonal_confusion_matrix.png')
    
    #print('Classification Report:')
    #print(classification_report(y_true, y_pred, labels=np.arange(n_class), target_names=artist_name))

#showClassficationReport_Generator(model, valid_generator, STEP_SIZE_VALID)
