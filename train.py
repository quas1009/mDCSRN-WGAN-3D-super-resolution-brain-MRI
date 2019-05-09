import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.keras.layers import Dense, Activation, Conv3D, ELU, Concatenate, BatchNormalization, Flatten, Input, Lambda
from tensorflow.keras import Model, metrics
from tensorflow.keras.constraints import Constraint
from tensorflow.keras import backend as K
import numpy as np
import nibabel as nib
import os
from skimage.util import view_as_windows
import scipy.io, time 
from sklearn.model_selection import train_test_split
import pandas as pd
from IPython import display
from matplotlib import pyplot as plt
tfe.enable_eager_execution()

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
        folder = os.path.join(self.DIR_PATH+str(idx))        
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
            lr_dataset = tf.data.Dataset.from_tensor_slices(self.extract_patches(lr_raw_data,58))
            hr_dataset = tf.data.Dataset.from_tensor_slices(self.extract_patches(hr_raw_data,58))
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

# ============== Define the BatchNormalization Layer in Keras Model ============= # 

class BN_ELU_Layer(tf.keras.layers.Layer):
    def __init__(self, is_training=True):
        super(BN_ELU_Layer, self).__init__()
        self.mark = is_training

    def call(self, input,is_training=True):
        a = tf.contrib.layers.batch_norm(input, self.mark)
        b = tf.nn.elu(a)        
        return b

# ================ Define the Generator Network Architecture =============== # 

class Generator(Model):
    # Generator variables all come from convolution layer
    def __init__(self):
        super(Generator, self).__init__(name='Init_Generator')
        kernel_size = 3
        growth_rate = 16        
        
        # -------- First Conv ---------  
        self.Input_conv = tf.keras.layers.Conv3D(2*growth_rate,
                                                  kernel_size,
                                                  strides=1,
                                                  use_bias = False,
                                                  padding='same')
        # -------- DenseBlock 1 ---------
        # L1
        self.DL11_bn_elu = BN_ELU_Layer()
        self.DL11_conv = tf.keras.layers.Conv3D(growth_rate, kernel_size,strides=1,
                                                use_bias = False, padding='same')
        # L2
        self.DL12_concat = tf.keras.layers.Concatenate() 
        self.DL12_bn_elu = BN_ELU_Layer()
        self.DL12_conv = tf.keras.layers.Conv3D(growth_rate, kernel_size,strides=1,
                                                use_bias = False, padding='same')
        # L3
        self.DL13_concat = tf.keras.layers.Concatenate() 
        self.DL13_bn_elu = BN_ELU_Layer()
        self.DL13_conv = tf.keras.layers.Conv3D(growth_rate, kernel_size,strides=1,
                                                use_bias = False, padding='same')
        # L4
        self.DL14_concat = tf.keras.layers.Concatenate() 
        self.DL14_bn_elu = BN_ELU_Layer()
        self.DL14_conv = tf.keras.layers.Conv3D(growth_rate, kernel_size,strides=1,
                                                use_bias = False, padding='same')
        # Concat
        self.DL15_concat = tf.keras.layers.Concatenate() 
        
        # -------- Compressor 1 ---------
        self.C1_concat = tf.keras.layers.Concatenate() 
        self.C1_conv = tf.keras.layers.Conv3D(2*growth_rate, 1,strides=1,
                                                use_bias = False, padding='same')
        
        # -------- DenseBlock 2 ---------
        # L1
        self.DL21_bn_elu = BN_ELU_Layer()
        self.DL21_conv = tf.keras.layers.Conv3D(growth_rate, kernel_size,strides=1,
                                                use_bias = False, padding='same') 
        # L2
        self.DL22_concat = tf.keras.layers.Concatenate() 
        self.DL22_bn_elu = BN_ELU_Layer()
        self.DL22_conv = tf.keras.layers.Conv3D(growth_rate, kernel_size,strides=1,
                                                use_bias = False, padding='same')
        # L3
        self.DL23_concat = tf.keras.layers.Concatenate() 
        self.DL23_bn_elu = BN_ELU_Layer()
        self.DL23_conv = tf.keras.layers.Conv3D(growth_rate, kernel_size,strides=1,
                                                use_bias = False, padding='same')
        # L4
        self.DL24_concat = tf.keras.layers.Concatenate() 
        self.DL24_bn_elu = BN_ELU_Layer()
        self.DL24_conv = tf.keras.layers.Conv3D(growth_rate, kernel_size,strides=1,
                                                use_bias = False, padding='same')
        # Concat
        self.DL25_concat = tf.keras.layers. Concatenate() 
        
        # -------- Compressor 2 ---------
        self.C2_concat = tf.keras.layers.Concatenate() 
        self.C2_conv = tf.keras.layers.Conv3D(2*growth_rate, 1,strides=1,
                                                use_bias = False, padding='same')
        
        # -------- DenseBlock 3 ---------
        # L1
        self.DL31_bn_elu = BN_ELU_Layer()
        self.DL31_conv = tf.keras.layers.Conv3D(growth_rate, kernel_size,strides=1,
                                                use_bias = False, padding='same') 
        # L2
        self.DL32_concat = tf.keras.layers.Concatenate() 
        self.DL32_bn_elu = BN_ELU_Layer()
        self.DL32_conv = tf.keras.layers.Conv3D(growth_rate, kernel_size,strides=1,
                                                use_bias = False, padding='same')
        # L3
        self.DL33_concat = tf.keras.layers.Concatenate() 
        self.DL33_bn_elu = BN_ELU_Layer()
        self.DL33_conv = tf.keras.layers.Conv3D(growth_rate, kernel_size,strides=1,
                                                use_bias = False, padding='same')
        # L4
        self.DL34_concat = tf.keras.layers.Concatenate() 
        self.DL34_bn_elu = BN_ELU_Layer()
        self.DL34_conv = tf.keras.layers.Conv3D(growth_rate, kernel_size,strides=1,
                                                use_bias = False, padding='same')
        # Concat
        self.DL35_concat = tf.keras.layers.Concatenate() 
        
        # -------- Compressor 3 ---------
        self.C3_concat = tf.keras.layers.Concatenate() 
        self.C3_conv = tf.keras.layers.Conv3D(2*growth_rate,1,strides=1,
                                                use_bias = False, padding='same')
        
        # -------- DenseBlock 4 ---------
        # L1
        self.DL41_bn_elu = BN_ELU_Layer()
        self.DL41_conv = tf.keras.layers.Conv3D(growth_rate, kernel_size,strides=1,
                                                use_bias = False, padding='same') 
        # L2
        self.DL42_concat = tf.keras.layers.Concatenate() 
        self.DL42_bn_elu = BN_ELU_Layer()
        self.DL42_conv = tf.keras.layers.Conv3D(growth_rate, kernel_size,strides=1,
                                                use_bias = False, padding='same')
        # L3
        self.DL43_concat = tf.keras.layers.Concatenate() 
        self.DL43_bn_elu = BN_ELU_Layer()
        self.DL43_conv = tf.keras.layers.Conv3D(growth_rate, kernel_size,strides=1,
                                                use_bias = False, padding='same')
        # L4
        self.DL44_concat = tf.keras.layers.Concatenate() 
        self.DL44_bn_elu = BN_ELU_Layer()
        self.DL44_conv = tf.keras.layers.Conv3D(growth_rate, kernel_size,strides=1,
                                                use_bias = False, padding='same')
        # Concat
        self.DL45_concat = tf.keras.layers.Concatenate() 
        
 
        # -------- Reconstruction ---------
        self.R_concat = tf.keras.layers.Concatenate() 
        self.R_conv = tf.keras.layers.Conv3D(1, 1,strides=1,
                                                use_bias = False, padding='same')
        
    

    def call(self, images, is_training=True):
        # -------- Conv ---------  
        result1 = self.Input_conv(images)
        # -------- DenseBlock 1 ---------
        # L1
        result11 = self.DL11_bn_elu(result1,is_training=is_training)
        result11 = self.DL11_conv(result11)
        # L2
        result_in = self.DL12_concat([result1,result11]) 
        result12 = self.DL12_bn_elu(result_in,is_training=is_training)
        result12 = self.DL12_conv(result12) 
        # L3
        result_in = self.DL13_concat ([result_in,result12])
        result13 = self.DL13_bn_elu(result_in,is_training=is_training)
        result13 = self.DL13_conv(result13)
        # L4
        result_in = self.DL14_concat([result_in,result13])
        result14 = self.DL14_bn_elu(result_in,is_training=is_training)
        result14 = self.DL14_conv(result14)
        # Concat
        result15 = self.DL15_concat([result_in,result14])
        
        # -------- Compressor 1 ---------
        result_c1 = self.C1_concat([result1,result15])
        result2 =  self.C1_conv(result_c1)
        
        # -------- DenseBlock 2 ---------
        # L1
        result21 = self.DL21_bn_elu(result2,is_training=is_training)
        result21 = self.DL21_conv(result21)
        # L2
        result_in = self.DL22_concat([result2,result21]) 
        result22 = self.DL22_bn_elu(result_in,is_training=is_training)
        result22 = self.DL22_conv(result22) 
        # L3
        result_in = self.DL23_concat([result_in,result22])
        result23 = self.DL23_bn_elu(result_in,is_training=is_training)
        result23 = self.DL23_conv(result23)
        # L4
        result_in = self.DL24_concat([result_in,result23])
        result24 = self.DL24_bn_elu(result_in,is_training=is_training)
        result24 = self.DL24_conv(result24)
        # Concat
        result25 = self.DL25_concat([result_in,result24])
        
        # -------- Compressor 2 ---------
        result_c2 = self.C2_concat([result_c1,result25])
        result3 =  self.C2_conv(result_c2)
        
        # -------- DenseBlock 3 ---------
        # L1
        result31 = self.DL31_bn_elu(result3,is_training=is_training)
        result31 = self.DL31_conv(result31)
        # L2
        result_in = self.DL32_concat([result3,result31]) 
        result32 = self.DL32_bn_elu(result_in,is_training=is_training)
        result32 = self.DL32_conv(result32) 
        # L3
        result_in = self.DL33_concat([result_in,result32])
        result33 = self.DL33_bn_elu(result_in,is_training=is_training)
        result33 = self.DL33_conv(result33)
        # L4
        result_in = self.DL34_concat([result_in,result33])
        result34 = self.DL34_bn_elu(result_in,is_training=is_training)
        result34 = self.DL34_conv(result34)
        # Concat
        result35 = self.DL35_concat([result_in,result34])
        
        # -------- Compressor 3 ---------
        result_c3 = self.C3_concat([result_c2,result35])
        result4 =  self.C3_conv(result_c3)
        
        # -------- DenseBlock 4 ---------
        # L1
        result41 = self.DL41_bn_elu(result4,is_training=is_training)
        result41 = self.DL41_conv(result41)
        # L2
        result_in = self.DL42_concat([result4,result41]) 
        result42 = self.DL42_bn_elu(result_in,is_training=is_training)
        result42 = self.DL42_conv(result42) 
        # L3
        result_in = self.DL43_concat([result_in,result42])
        result43 = self.DL43_bn_elu(result_in,is_training=is_training)
        result43 = self.DL43_conv(result43)
        # L4
        result_in = self.DL44_concat([result_in,result43])
        result44 = self.DL44_bn_elu(result_in,is_training=is_training)
        result44 = self.DL44_conv(result44)
        # Concat
        result45 = self.DL45_concat([result_in,result44])

        # -------- Reconstruction ---------
        result5 = self.R_concat([result_c3,result45])
        result6 = self.R_conv(result5)
        
        return result6

# ============== Define the LayerNormalization Layer in Keras Model ============= # 

class LN_LRELU_Layer(tf.keras.layers.Layer):
    def __init__(self, is_training=True):
        super(LN_LRELU_Layer, self).__init__()
        self.mark = is_training

    def call(self, input,is_training=True):
        # Override call() instead of __call__ so we can perform some bookkeeping.
        a = tf.contrib.layers.layer_norm(input, self.mark)
        b = tf.nn.leaky_relu(a,0.2)        
        return b

# ============== Weights Clip ============= # 
class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''
    def __init__(self, c=2):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}

# ================ Define the Discriminator Network Architecture =============== # 
class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        kernel_size = 3
        growth_rate = 64
        self.lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)
        
        # ----------------L1: Conv + LRelu (Expand Channels) -----------------------
        
        self.conv1 = tf.keras.layers.Conv3D(growth_rate,
                                            kernel_size,
                                            strides=1,
                                            activation=self.lrelu,
                                            kernel_constraint = WeightClip(0.01),
                                            padding='same')
        
        # ----------------L2: Conv + LN + LRelu (Reduce FOV) -----------------------
        self.conv2_rf = tf.keras.layers.Conv3D(growth_rate,
                                               kernel_size,
                                               strides=(2,2,2),
                                               kernel_constraint = WeightClip(0.01),
                                               padding='same')
        self.L2LN = LN_LRELU_Layer()
        
        # ---------------- C1:ConvStride Block 1 -----------------------
        self.conv3_ec = tf.keras.layers.Conv3D(2*growth_rate,
                                               kernel_size,strides=1,
                                               kernel_constraint = WeightClip(0.01),
                                               padding='same') #expand channels
        self.C1LN1 = LN_LRELU_Layer()

        self.conv3_rf = tf.keras.layers.Conv3D(2*growth_rate,
                                               kernel_size,strides=(2,2,2),
                                               kernel_constraint = WeightClip(0.01),
                                               padding='same')
        self.C1LN2 = LN_LRELU_Layer()
        
        # ---------------- C2: ConvStride Block 2 -----------------------
        self.conv4_ec = tf.keras.layers.Conv3D(4*growth_rate,
                                               kernel_size,strides=1,
                                               kernel_constraint = WeightClip(0.01),
                                                padding='same')
        self.C2LN1 = LN_LRELU_Layer()
        self.conv4_rf = tf.keras.layers.Conv3D(4*growth_rate,
                                               kernel_size,strides=(2,2,2),
                                               kernel_constraint = WeightClip(0.01),
                                               padding='same')
        self.C2LN2 = LN_LRELU_Layer()
        
        # ---------------- C3:ConvStride Block 3 -----------------------
        self.conv5_ec = tf.keras.layers.Conv3D(8*growth_rate,
                                               kernel_size,strides=1,
                                               kernel_constraint = WeightClip(0.01),
                                               padding='same')
        self.C3LN1 = LN_LRELU_Layer()
        self.conv5_rf = tf.keras.layers.Conv3D(8*growth_rate,
                                               kernel_size,strides=(2,2,2),
                                               kernel_constraint = WeightClip(0.01),
                                               padding='same')
        self.C3LN2 = LN_LRELU_Layer()
        
        # ----------------L3: FC + L-ReLu + FC -------------------
        self.L3_flat = tf.keras.layers.Flatten()
        self.L3_fc1   = tf.keras.layers.Dense(1024,activation=self.lrelu,kernel_constraint = WeightClip(0.01))
        self.L3_fc2  = tf.keras.layers.Dense(1,kernel_constraint = WeightClip(0.01))
        
          
    def call(self, images,is_training=True):
        # ----------------Conv + LRelu (Expand Channels) -----------------------
        result1 = self.conv1 (images)
        # ----------------Conv + LN + LRelu (Reduce FOV) -----------------------
        result2 = self.conv2_rf(result1)
        result2 = self.L2LN(result2,is_training=is_training)
        
        # ---------------- ConvStride Block 1 -----------------------
        result3 = self.conv3_ec(result2)
        result3 = self.C1LN1(result3,is_training=is_training)
        result3 = self.conv3_rf(result3)
        result3 = self.C1LN2(result3,is_training=is_training)
        
        # ---------------- ConvStride Block 2 -----------------------
        result4 = self.conv4_ec(result3)
        result4 = self.C2LN1(result4,is_training=is_training)
        result4 = self.conv4_rf(result4)
        result4 = self.C2LN2(result4,is_training=is_training)
        
        # ---------------- ConvStride Block 3 -----------------------
        result5 = self.conv5_ec(result4)
        result5 = self.C3LN1(result5,is_training=is_training)
        result5 = self.conv5_rf(result5)
        result5 = self.C3LN2(result5,is_training=is_training)
        
        # ---------------- FC + L-ReLu + FC -------------------
        result6 = self.L3_flat(result5)
        result6 = self.L3_fc1(result6)
        result6 = self.L3_fc2 (result6)
        
        return result6


train_set, test_set, val_set, eval_set = load_idset('filelist.csv') 
# ------------------ get one sample -------------------------------
sample_id = val_set.take(1)
sample_indices=np.array([idx.numpy() for a, idx in enumerate(sample_id)])
Sampleloader = DatafromCSV(sample_indices)
sample_set = Sampleloader.load_patchset()

# =========================== Toolbox ============================ #

def disp(sample_set,cube,step_counter):
    sample_set=sample_set.batch(2)
    PRED = np.array([], dtype=np.float32).reshape([0,cube,cube,cube,1])
    HR = np.array([], dtype=np.float32).reshape([0,cube,cube,cube,1])
    for idx, (lr, hr) in enumerate(sample_set):
        pred = np.array(generator.predict(lr.numpy()))
        pred = np.array(pred,dtype = np.float32)
        PRED = np.concatenate((PRED,pred))
        hr = np.array(hr,dtype = np.float32)
        HR = np.concatenate((HR,hr))
    inp = merge(PRED, cube, 1)
    tar = merge(HR, cube, 1)
    
    plt.figure(figsize=(5,5))
    plt.subplot(1,2,1)
    plt.imshow(np.squeeze(inp[0,63,:,:,:]))
    plt.title('predict')
    plt.subplot(1,2,2)
    plt.imshow(np.squeeze(tar[0,63,:,:,:]))
    plt.title('truth')
    plt.axis('off')
    filename = './figure/'+str(step_counter)+'.png'
    Handle = plt.gcf()
    Handle.savefig(filename) 
    plt.show()
    
    return inp, tar

def merge(concated_tensor,cube,x):
    d1 = int(256/cube)
    d2 = int(320/cube)
    d3 = int(cube*cube*cube)
    e = np.reshape(concated_tensor,[x,d1,d2,d2,d3])
    f1 = np.split(e,x,axis=0)
    f2=[]
    for item in f1:
        f2.extend(np.split(item,d1,axis=1))
    f3 = []
    for item in f2:
        f3.extend(np.split(item,d2,axis=2))
    f = []
    for item in f3:
        f.extend(np.split(item,d2,axis=3))
    g = [np.reshape(item,[1,cube,cube,cube,1]) for item in f]
    h1 = [ np.concatenate(g[d2*i: d2*(i+1)], axis = 3) for i in range(d1*d2*x) ]
    h2 = [ np.concatenate(h1[d2*i: d2*(i+1)], axis = 2) for i in range(d1*x) ]
    h3 = [ np.concatenate(h2[d1*i: d1*(i+1)], axis = 1) for i in range(x) ]
    h = np.concatenate(h3, axis = 0)
    return h


def score_patch(pred_patch, true_patch, c):
    pred_patch = np.squeeze(pred_patch[0,:,:,:,:])
    true_patch = np.squeeze(true_patch[0,:,:,:,:])
    pred_cs = np.squeeze(pred_patch[c,:,:])
    pred_cs = tf.expand_dims(pred_cs,-1)
    true_cs = np.squeeze(true_patch[c,:,:])
    true_cs = tf.expand_dims(true_cs,-1)
    
    ssim = tf.image.ssim(pred_cs, true_cs, max_val=1.0)
    psnr = tf.image.psnr(pred_patch, true_patch, max_val=1.0)
    mse = tf.losses.mean_squared_error(pred_patch,true_patch)
    print('----------------------------------')
    print('SSIM:{} '.format(ssim.numpy()))
    print('PSNR:{} '.format(psnr.numpy()))
    print('MSE:{} '.format(mse.numpy()))
    print('----------------------------------')

def DLoss(y_true,y_pred): 
  d_loss =  tf.reduce_mean(y_pred) - tf.reduce_mean(y_true)
  return d_loss

# ========================== Formal Train =========================== # 
#------------------------ Hyper parameters ------------------- #
batch_size = 2
patch_size = 2
n_epochs = 10
BUFFER_SIZE = 777
n_patches = np.ceil(256/64)*np.ceil(320/64)*np.ceil(320/64)*batch_size

#-------------------------- Optimizer ----------------------- #

g_optimizer = tf.train.RMSPropOptimizer(0.0001)
d_optimizer = tf.train.RMSPropOptimizer(0.0001)
discriminator.compile(loss=DLoss,
                  optimizer=d_optimizer ,
                  metrics =['mse'])


#-------------------------- Checkpoint ----------------------- #
# Works On Colab
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_wgan")
checkpoint = tf.train.Checkpoint(g_optimizer=g_optimizer,
                                 d_optimizer=d_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

#--------------- Marker for Loops Initialization ------------- #
batch_counter = 0
step_counter = 0 # control step for 
d_loss_history = []
g_loss_history = []


flag_only_D = True # flag for training Discriminator only, for the first 10k steps.
flag_G = False
# ================ Formal Training Process ================ #
'''
Basics: 
- Use pretrained generator: 
1) Train discriminator for 10k steps
2) D:G = 7:1, for 500 steps
3) D: extra 200 steps

2) + 3) for 550k steps totally

'''

for epoch in range(n_epochs):
    start = time.time()
    # ------------------- Split batch in idset ---------------------
    train_set = train_set.shuffle(buffer_size=BUFFER_SIZE).batch(batch_size)
    iter = train_set.make_one_shot_iterator()
    for el in iter:
        batch_counter += 1
        
        indices=np.array(el)
    # ---------------------- Get data from id ----------------------
        Batchloader = DatafromCSV(indices)
        train_dataset = Batchloader.load_patchset()
    # -------------------- Split patch in dataset ------------------
        train_dataset = train_dataset.shuffle(buffer_size=n_patches).batch(patch_size)
        for patch, (lr, hr) in enumerate(train_dataset): 
            step_counter += 1
            
            # train only discriminator for first 10k steps
            # from 10001 steps, reset step_counter and enter the formal loop
            flag_G = False
            if flag_only_D:
                if step_counter == 10001:
                  step_counter = 1
                  flag_only_D = False
                else:
                  print('Step for discriminator training:{}'.format(step_counter))              
           
            if flag_only_D == False:
                if step_counter % 700 >= 500:
                    flag_G = False
                elif step_counter % 8 < 7:
                    flag_G = False
                else:
                    flag_G = True

                print('Global Step:{}, Subject No.{}, is training on G?: {}'.format(step_counter,batch_counter, flag_G))
            
            # training for generator
            if flag_G:
                with tf.GradientTape() as g_tape:

                    g_output = generator(lr, is_training = True)
                    d_fake_output = discriminator.predict(g_output.numpy())
                    g_vars = generator.variables
                    
                    mse_loss = tf.losses.mean_squared_error(hr, g_output) #L2 Loss for Generator(Parts)
                    g_gan_loss = - 1e-3 * tf.reduce_mean(d_fake_output)
                    g_loss = mse_loss + g_gan_loss
                    g_loss_history.append(g_loss)
                    
                    
                    with g_tape.stop_recording():
                        g_gradients = g_tape.gradient(g_loss,g_vars)  #generator.variables = g_vars
                        g_optimizer.apply_gradients(zip(g_gradients,g_vars))
                    print('Generator Loss:{}'.format(g_loss))
            else:
                    if step_counter == 1 and flag_only_D :
                        temp = generator(tf.random.uniform([1,64,64,64,1]),is_training=False)
                        g_output = generator.predict(lr.numpy())
                        d_real_output = np.array([2,1])
                    else:
                          g_output = generator.predict(lr.numpy())
                          d_real_output = discriminator.predict(hr.numpy())
                    
                    d_loss,_ = discriminator.train_on_batch(g_output,d_real_output,reset_metrics=False)
                    d_loss_history.append(d_loss)
                    print('Discriminator Loss:{}'.format(d_loss))
                         
            if step_counter % 200 == 0 and flag_only_D == False:
                manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=100)
                save_path = manager.save()
                print("\n----------Saved checkpoint for step {}: {}-------------\n".format(step_counter, save_path))
                          
                f=open('g_loss.txt','a')
                for g in g_loss_history:
                  f.write(str(g))
                  f.write('\n')
                f.close()
                g_loss_history = []
        
                f=open('h_loss.txt','a')
                for h in h_loss_history:
                  f.write(str(h))
                  f.write('\n')
                f.close()
                h_loss_history = []
             
            if step_counter %  301 ==0 and flag_only_D == False:
                #export evaluating parameters for [32,:,:] in a current patch
                score_patch(generator.predict(lr.numpy()), hr, 32)  
        
            if step_counter % 701 == 0 and flag_only_D == False:
                display.clear_output(wait=True)

            if step_counter == 55000:
                print('\n ----------------- Completed for 55k steps! --------------------------\n')
                
                break
        if step_counter == 55000:
          break
    if step_counter == 55000:
      break
              
