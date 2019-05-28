import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.keras import Model, metrics
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Activation, Conv3D, ELU, Concatenate
from tensorflow.keras.layers import BatchNormalization, Flatten, Input, Lambda
from tensorflow.keras.constraints import Constraint

import numpy as np
import os, scipy, time
from IPython import display

from dataset import DatafromCSV, load_idset
from utils import disp, merge, score_patch

tfe.enable_eager_execution()

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
    
train_set, test_set, val_set, eval_set = load_idset('filelist.csv') 
# ------------------ get one sample -------------------------------
sample_id = val_set.take(1)
sample_indices=np.array([idx.numpy() for a, idx in enumerate(sample_id)])
Sampleloader = DatafromCSV(sample_indices)
sample_set = Sampleloader.load_patchset()

# ========================== Initial Train =========================== # 
# ----- The Loss is need further validation: from valid images --------# 
#------------------------ Hyper parameters ------------------- #
batch_size = 2
patch_size = 2
n_epochs = 10
BUFFER_SIZE = 777
n_patches = np.ceil(256/64)*np.ceil(320/64)*np.ceil(320/64)*batch_size
learning_rate = 1e-4 
beta1 = 0.5

#-------------------------- Optimizer ----------------------- #
g_optimizer_init = tf.train.AdamOptimizer(learning_rate,beta1=beta1)

generator = Generator() 
generator.compile(loss='mean_absolute_error',
                  optimizer=g_optimizer_init ,
                  metrics =['mse'])


#-------------------------- Checkpoint ----------------------- #
# Works On Colab
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(g_optimizer_init=g_optimizer_init,
                                 generator=generator)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
#--------------- Marker for Loops Initialization ------------- #
batch_counter = 0
step_counter = 0 
g_loss_history = []
valid_loss_history = []

for epoch in range(n_epochs):
    start = time.time()
    # ------------------- Split batch in idset ---------------------
    train_set = train_set.shuffle(buffer_size=BUFFER_SIZE).batch(batch_size)
    iter = train_set.make_one_shot_iterator()
    for el in iter:
        batch_counter += 1
        display.clear_output(wait=True)
        indices=np.array(el)
    # ---------------------- Get data from id ----------------------
        Batchloader = DatafromCSV(indices)
        train_dataset = Batchloader.load_patchset()
    # -------------------- Split patch in dataset ------------------
        train_dataset = train_dataset.shuffle(buffer_size=n_patches).batch(patch_size)
        for patch, (lr, hr) in enumerate(train_dataset): 
            step_counter += 1
            print('Global Step:{}, Subject No.{}'.format(step_counter,batch_counter))
            g_loss,_ = generator.train_on_batch(lr,hr,reset_metrics=False)
            g_loss_history.append(g_loss);            

            if step_counter % 90 == 0:
                manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=100)
                save_path = manager.save()
                print("\n----------Saved checkpoint for step {}: {}-------------\n".format(step_counter, save_path))
            if step_counter == 50000:
                print('\n ----------------- Completed for 50k steps! --------------------------\n')
                break
        if step_counter == 50000:
          break
    if step_counter == 50000:
      break