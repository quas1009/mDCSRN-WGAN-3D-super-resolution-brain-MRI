import tensorflow as tf
import numpy as np
import pandas as pd
import nibabel as nib
import os, scipy
from sklearn.model_selection import train_test_split


# ================ Load Data from subject list =============== # 

class DatafromCSV(object):
    def __init__(self, indices):
        self.indices = indices
        self.length = indices.shape[0]
        self.DIR_PATH = 'mnt/superresolution/HCP_1200/'
        self.SUB_DIR = '/unprocessed/3T/T1w_MPR1'
        self.CUBE = 64
        
    def transform(self, image):
        image = np.array(image/4095)
        data = tf.convert_to_tensor(image, dtype=np.float32)
        return data
        
    def get_item(self,idx):
        mat = os.path.join(self.DIR_PATH + str(idx)+ self.SUB_DIR, str(idx)+'_LR.mat')
        nii = os.path.join(self.DIR_PATH + str(idx)+ self.SUB_DIR, str(idx)+'_3T_T1w_MPR1.nii.gz')
        
        lr = scipy.io.loadmat(mat)['out_final']
        hr = nib.load(nii).get_fdata()
        lr_sampler = self.transform(lr)
        hr_sampler = self.transform(hr)
        return (lr_sampler, hr_sampler)
      
    def get_data(self):
        lr_data = np.zeros([self.length,256,320,320], dtype=np.float32)
        hr_data = np.zeros([self.length,256,320,320], dtype=np.float32)

        for i in range(self.length):
            idx = self.indices[i][0]
            (lr_data[i,:,:,:] , hr_data[i,:,:,:]) = self.get_item(idx)
        return (lr_data, hr_data)
      
    def extract_patches(self,data_4d, stride):
        cube = self.CUBE
        data_5d = tf.expand_dims(data_4d,-1)
        patches=tf.extract_volume_patches(
            input=data_5d,
            ksizes=[1, cube, cube, cube, 1],
            strides=[1, stride, stride, stride, 1],
            padding='VALID',
        )
        result_tf = tf.reshape(patches, [-1, cube, cube, cube])
        img = result_tf
        result = tf.expand_dims(img,-1)
        return result
      
    def load_patchset(self,is_train = True):
        lr_raw_data, hr_raw_data = self.get_data()
        if is_train:
            lr_dataset = tf.data.Dataset.from_tensor_slices(self.extract_patches(lr_raw_data,64))
            hr_dataset = tf.data.Dataset.from_tensor_slices(self.extract_patches(hr_raw_data,64))
            dataset = tf.data.Dataset.zip((lr_dataset, hr_dataset))
        else:
            paddings = tf.constant([[0, 0],[40, 0], [34, 0], [34,0]])
            lr_data = tf.pad(lr_raw_data, paddings, "CONSTANT")
            hr_data = tf.pad(hr_raw_data, paddings, "CONSTANT")
            lr_dataset = tf.data.Dataset.from_tensor_slices(self.extract_patches(lr_data,58))
            hr_dataset = tf.data.Dataset.from_tensor_slices(self.extract_patches(hr_data,58))
            dataset = tf.data.Dataset.zip((lr_dataset, hr_dataset))
        return dataset

# ====================== Load and split dataset ====================== # 

def load_idset(csv_name):
    
    csv_path = os.path.join(csv_name)
    filelist = pd.read_csv(csv_path)
    
    train_idset,test_idset = train_test_split(filelist, test_size=0.2)
    train_idset,val_idset_ = train_test_split(train_idset, test_size=0.125)
    test_idset,eval_idset_ = train_test_split(test_idset, test_size=0.5)
    print(len(train_idset),len(test_idset),len(val_idset_),len(eval_idset_))

    train_set = tf.data.Dataset.from_tensor_slices(np.array(train_idset))
    test_set = tf.data.Dataset.from_tensor_slices(np.array(test_idset))
    val_set = tf.data.Dataset.from_tensor_slices(np.array(val_idset_))
    eval_set = tf.data.Dataset.from_tensor_slices(np.array(eval_idset_))
    
    return train_set, test_set, val_set, eval_set

