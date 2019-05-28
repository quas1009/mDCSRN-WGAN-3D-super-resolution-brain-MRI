from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

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
