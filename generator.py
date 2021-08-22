from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

import numpy as np
from collections import Counter
import tensorflow as tf
import tensorflow.keras
import cv2

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x
def DataAugmentation3(image):
  n = 8
  im_list = []
  iv_list = []
  patch_initial = np.array([0,0])
  patch_scale = 1/n #find 5 patch on diagonal
  smaller_dim = np.min(image.shape[0:2])
  #print(smaller_dim)
  image = cv2.resize(image,((smaller_dim,smaller_dim)))
  patch_size = int(patch_scale * smaller_dim)
  #print(patch_size)
  for i in range(n):
      patch_x = patch_initial[0]
      patch_y = patch_initial[1]
      patch_image = image[patch_x:patch_x+patch_size,patch_y:patch_y+patch_size]
      #print(patch_image.shape)
      #patch_image = zoomin(patch_image,3)
      #print(patch_image.shape)
      x2 = smaller_dim - patch_x
      patch_image2 = image[x2-patch_size:x2,patch_y:patch_y+patch_size]
      #patch_image2 = zoomin(patch_image2,3)
      patch_initial = np.array([patch_x+patch_size,patch_y+patch_size])
      iv_list.append(patch_image)
      im_list.append(patch_image2)
  im_list = im_list[1:n]
  im_h = cv2.vconcat(iv_list)
  #print(im_h.shape)
  width = patch_size*(n-1)
  #print(width)
  image = cv2.resize(image,(width,width))
  im_v=cv2.hconcat(im_list)
  #print(im_v.shape)
  im_v = cv2.vconcat([image,im_v])
  #print(im_v.shape)
  img = cv2.hconcat([im_v,im_h])
  return img

def get_random_augment(x,params):
  img_row_axis = 0
  img_col_axis = 1
  
  if params['augment_painting']:
    x=DataAugmentation3(x)
  else:
    if params['horizontal_flip']:
      x = flip_axis(x, img_col_axis)

    if params['vertical_flip']:
      x = flip_axis(x, img_row_axis)
  return x

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

def generate_classdict2(list):
    label=list(itertools.chain(*list))
    generate_classdict(label)

class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, df, 
                 batch_size,is_train=False,path='./images/',
                 input_size=(224, 224, 3),
                 shuffle=True):
        
        self.df = df.copy()
        # self.X_col = X_col
        # self.y_col = y_col
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.path = path
        self.horizontal_flip=True
        self.vertical_flip=True
        self.augment_painting=False
        self.is_train=is_train
        
        self.n = len(self.df)
        self.Artist_class_num,self.Artist_class_list,self.Artist_class_dict,self.Artist_class_weight=generate_classdict(np.array(df['Artist'].tolist()) )
        self.Style_class_num,self.Style_class_list,self.Style_class_dict,self.Style_class_weight=generate_classdict(np.array(df['Style'].tolist()))
        self.Objtype_class_num,self.Objtype_class_list,self.Objtype_class_dict,self.Objtype_class_weight=generate_classdict(np.array(df['Object Type'].tolist()) )
        self.CreationDate_class_num, self.CreationDate_class_list,self.CreationDate_class_dict,self.CreationDate_class_weight=generate_classdict(np.array(dataset['Creation Date'].tolist()))
        # self.n_name = df[y_col['name']].nunique()
        # self.n_type = df[y_col['type']].nunique()
        # self.on_epoch_end()
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __getitem__(self, index):

        if (index+1)*self.batch_size<=self.n:
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
            num=self.batch_size
        else:
            indexes = self.indexes[index*self.batch_size:]
            num = self.n - index*self.batch_size

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp,num)

        return X, y
    
    def __len__(self):
        return self.n // self.batch_size

    def __data_generation(self,list_IDs_temp,num):

        x_array=np.zeros((num,self.input_size[0],self.input_size[1]))
        y1_array=[]
        y2_array=[]
        y3_array=[]
        y4_array=[]
        count=0

        for i in list_IDs_temp:
            if self.is_train:
                flip_horizontal = (np.random.random() < 0.5) * self.horizontal_flip
                flip_vertical = (np.random.random() > 0.5) * self.vertical_flip
                painting_augment = (np.random.random() > 0.5) * self.augment_painting
                params={
                    'horizontal_flip':flip_horizontal,
                    'vertical_flip':flip_vertical,
                    'augment_painting':painting_augment,
                }
            img=(img_to_array(load_img(self.path+self.df[i]['imagefile'],target_size=self.input_size))*1./255)
            if self.is_train:
                img=get_random_augment(img,params)
            x_array[count]=img
            y1_array.append(to_categorical(self.Artist_class_dict[self.df[i]['Artist']],num_classes=self.Artist_class_num))
            y2_array.append(to_categorical(self.Style_class_dict[self.df[i]['Style']],num_classes=self.Style_class_num))
            y3_array.append(to_categorical(self.Objtype_class_dict[self.df[i]['Object Type']],num_classes=self.Objtype_class_num))
            y4_array.append(to_categorical(self.CreationDate_class_dict[self.df[i]['Creation Date']],num_classes=self.CreationDate_class_num))
            count+=1
        return x_array,{'Artist_output':y1_array,'Style_output':y2_array,'Objtype_output':y3_array,'CreationDate_output':y4_array}


