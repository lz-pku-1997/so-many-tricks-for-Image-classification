

# %% [code]
bool_random_flip_left_right=1
bool_random_flip_up_down=1
bool_random_brightness=0
bool_random_contrast=0
bool_random_hue=0
bool_random_saturation=0
bool_random_crop = 0

bool_rotation_transform=1
cutmix_rate=0.
mixup_rate= 0.
gridmask_rate = 0.5

pre_trained='noisy-student' # None,'imagenet','noisy-student'
dense_activation='softmax' #'softmax','sigmoid'
bool_lr_scheduler=1

# 交叉验证和tta目前最多只允许使用一个
tta_times=0 #15 #当tta_times=i>0时，使用i+1倍测试集
cross_validation_folds=5 #当tta_times=i>1时，使用i折交叉验证


#focal_loss和label_smoothing最多同时使用一个
bool_focal_loss = 1
label_smoothing_rate=0.

#伪标签
bool_pseudo=0

# 增加auc记录
special_monitor='auc'

# %% [code]
#针对前面opts的一些中间处理

#tta只有在使用了data_aug时才允许启用
bool_tta =  tta_times and max(  bool_random_flip_left_right,
                                bool_random_flip_up_down,
                                bool_random_brightness,
                                bool_random_contrast,
                                bool_random_hue,
                                bool_random_saturation)

print(bool_tta)

assert (bool_focal_loss and label_smoothing_rate) == 0 , 'focal_loss和label_smoothing最多同时使用一个'
assert (tta_times and cross_validation_folds) == 0 , 'focal_loss和label_smoothing最多同时使用一个'


# %% [code]
#安装包
!pip install -U efficientnet
!pip install tensorflow_addons

# %% [code]
#导入包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os
import tensorflow as tf
import random, re, math
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
import tensorflow_addons as tfa

from kaggle_datasets import KaggleDatasets
import efficientnet.tfkeras as efn
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import ModelCheckpoint

print(tf.__version__)
print(tf.keras.__version__)

# %% [code]
#针对不同硬件产生不同配置
AUTO = tf.data.experimental.AUTOTUNE
# Detect hardware, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    GCS_DS_PATH = KaggleDatasets().get_gcs_path()
    print(GCS_DS_PATH)
else:
    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.
print("REPLICAS: ", strategy.num_replicas_in_sync)

# %% [code]
#path、label都作为全局变量

#超参数，根据数据和策略调参
BATCH_SIZE = 1 * strategy.num_replicas_in_sync
img_size = 768
EPOCHS = 25
lr_if_without_scheduler = 0.0003
nb_classes = 3
print('BATCH_SIZE是：',BATCH_SIZE)

my_metrics = ['accuracy']



import glob
train_paths = []
train_labels = []
test_paths = []
test_labels = []


# path='../input/small-big-culture/Candida'
# img=glob.glob(path+'/*')
# for i in range(len(img)):
#     train_paths.append(img[i])
#     train_labels.append([1,0,0,0])

    
# path='../input/small-big-culture/Microsporum'
# img=glob.glob(path+'/*')
# for i in range(len(img)):
#     train_paths.append(img[i])
#     train_labels.append([0,1,0,0])

    
# path='../input/small-big-culture/Trichophyton'
# img=glob.glob(path+'/*')
# for i in range(len(img)):
#     train_paths.append(img[i])
#     train_labels.append([0,0,1,0])

    
# path='../input/small-big-culture/Trichosporon'
# img=glob.glob(path+'/*')
# for i in range(len(img)):
#     train_paths.append(img[i])
#     train_labels.append([0,0,0,1])
    

    
paths = os.listdir('../input/slide-culture')
for i in range(3):
    img_dir = '../input/slide-culture/'+paths[i]
    img = os.listdir(img_dir)
    for ele in img:
        train_paths.append(img_dir+'/'+ele)     
        train_labels.append(np.eye(3)[i])
    
    

    
train_labels=np.array(train_labels)



    

# print((train_paths))
# print((train_labels))    
    
print(len(train_paths))
print(len(train_labels))
print(len(test_paths))

print('\n',train_labels[0])


from matplotlib import pyplot as plt
#随便看一张
img = plt.imread(train_paths[0])
print('\n',img.shape)
plt.imshow(img)

# %% [code]
#lr_scheduler
#数值按实际情况设置

LR_START = 0.00001
LR_MAX = 0.0001 * strategy.num_replicas_in_sync
LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 3
LR_SUSTAIN_EPOCHS = 5
LR_EXP_DECAY = .8

def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr
    
lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

rng = [i for i in range(EPOCHS)]
y = [lrfn(x) for x in rng]
if bool_lr_scheduler:
    plt.plot(rng, y)
    print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))

# %% [code]
def decode_image(filename, label=None,image_size=(img_size, img_size)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    
    if label is None:
        return image
    else:
        return image, label





# 只能写入test data 也能用的 aug   
def data_aug(image, label=None, image_size=[img_size, img_size]):
    if bool_random_flip_left_right:
        image = tf.image.random_flip_left_right(image)
    if bool_random_flip_up_down:    
        image = tf.image.random_flip_up_down(image)
    if bool_random_brightness:
        image = tf.image.random_brightness(image,0.2)
    if bool_random_contrast:
        image = tf.image.random_contrast(image,0.6,1.4)
    if bool_random_hue:
        image = tf.image.random_hue(image,0.07)
    if bool_random_saturation:
        image = tf.image.random_saturation(image,0.5,1.5)
    if bool_random_crop:
        image = tf.image.resize(image, [int(1.2*ele) for ele in image_size])
        image = tf.image.random_crop(image,image_size+[3])
    

    
    if label is None:
        return image
    else:
        return image, label

# %% [code]
import tensorflow as tf, tensorflow.keras.backend as K
def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    # returns 3x3 transformmatrix which transforms indicies
        
    # CONVERT DEGREES TO RADIANS
    rotation = math.pi * rotation / 180.
    shear = math.pi * shear / 180.
    
    # ROTATION MATRIX
    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')
    rotation_matrix = tf.reshape( tf.concat([c1,s1,zero, -s1,c1,zero, zero,zero,one],axis=0),[3,3] )
        
    # SHEAR MATRIX
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    shear_matrix = tf.reshape( tf.concat([one,s2,zero, zero,c2,zero, zero,zero,one],axis=0),[3,3] )    
    
    # ZOOM MATRIX
    zoom_matrix = tf.reshape( tf.concat([one/height_zoom,zero,zero, zero,one/width_zoom,zero, zero,zero,one],axis=0),[3,3] )
    
    # SHIFT MATRIX
    shift_matrix = tf.reshape( tf.concat([one,zero,height_shift, zero,one,width_shift, zero,zero,one],axis=0),[3,3] )
    
    return K.dot(K.dot(rotation_matrix, shear_matrix), K.dot(zoom_matrix, shift_matrix))



def rotation_transform(image,label):
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image randomly rotated, sheared, zoomed, and shifted
    DIM = img_size
    XDIM = DIM%2 #fix for size 331
    
    rot = 15. * tf.random.normal([1],dtype='float32')
    shr = 5. * tf.random.normal([1],dtype='float32') 
    h_zoom = 1.0 + tf.random.normal([1],dtype='float32')/10.
    w_zoom = 1.0 + tf.random.normal([1],dtype='float32')/10.
    h_shift = 16. * tf.random.normal([1],dtype='float32') 
    w_shift = 16. * tf.random.normal([1],dtype='float32') 
  
    # GET TRANSFORMATION MATRIX
    m = get_mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift) 

    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )
    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )
    z = tf.ones([DIM*DIM],dtype='int32')
    idx = tf.stack( [x,y,z] )
    
    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(m,tf.cast(idx,dtype='float32'))
    idx2 = K.cast(idx2,dtype='int32')
    idx2 = K.clip(idx2,-DIM//2+XDIM+1,DIM//2)
    
    # FIND ORIGIN PIXEL VALUES           
    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )
    d = tf.gather_nd(image,tf.transpose(idx3))
        
    return tf.reshape(d,[DIM,DIM,3]),label

# %% [code]
# 在batch内部互相随机取图
def cutmix(image, label, PROBABILITY = cutmix_rate):
    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
    # output - a batch of images with cutmix applied
    label=tf.cast(label,tf.float32)
    
    DIM = img_size    
    imgs = []; labs = []
    
    for j in range(BATCH_SIZE):
        
        #random_uniform( shape, minval=0, maxval=None)        
        # DO CUTMIX WITH PROBABILITY DEFINED ABOVE
        P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.int32)
        
        # CHOOSE RANDOM IMAGE TO CUTMIX WITH
        k = tf.cast(tf.random.uniform([], 0, BATCH_SIZE), tf.int32)
        
        # CHOOSE RANDOM LOCATION
        #选一个随机的中心点
        x = tf.cast(tf.random.uniform([], 0, DIM), tf.int32)
        y = tf.cast(tf.random.uniform([], 0, DIM), tf.int32)
        
        # Beta(1, 1)等于均匀分布
        b = tf.random.uniform([], 0, 1) # this is beta dist with alpha=1.0
        
        #P只随机出0或1，就是裁剪或是不裁剪
        WIDTH = tf.cast(DIM * tf.math.sqrt(1-b),tf.int32) * P
        ya = tf.math.maximum(0,y-WIDTH//2)
        yb = tf.math.minimum(DIM,y+WIDTH//2)
        xa = tf.math.maximum(0,x-WIDTH//2)
        xb = tf.math.minimum(DIM,x+WIDTH//2)
        
        # MAKE CUTMIX IMAGE
        one = image[j,ya:yb,0:xa,:]
        two = image[k,ya:yb,xa:xb,:]
        three = image[j,ya:yb,xb:DIM,:]        
        #得出了ya:yb区间内的输出图像
        middle = tf.concat([one,two,three],axis=1)
        #得到了完整输出图像
        img = tf.concat([image[j,0:ya,:,:],middle,image[j,yb:DIM,:,:]],axis=0)
        imgs.append(img)
        
        # MAKE CUTMIX LABEL
        #按面积来加权的
        a = tf.cast(WIDTH*WIDTH/DIM/DIM,tf.float32)
        lab1 = label[j,]
        lab2 = label[k,]
        labs.append((1-a)*lab1 + a*lab2)

    image2 = tf.reshape(tf.stack(imgs),(BATCH_SIZE,DIM,DIM,3))
    label2 = tf.reshape(tf.stack(labs),(BATCH_SIZE, nb_classes))
    return image2,label2

# %% [code]
def mixup(image, label, PROBABILITY = mixup_rate):
    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
    # output - a batch of images with mixup applied
    DIM = img_size
    
    imgs = []; labs = []
    for j in range(BATCH_SIZE):
        
        # CHOOSE RANDOM
        k = tf.cast( tf.random.uniform([],0,BATCH_SIZE),tf.int32)
        a = tf.random.uniform([],0,1) # this is beta dist with alpha=1.0

        #根据概率抽取执不执行mixup
        P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.int32)
        if P==1:
            a=0.
        
        # MAKE MIXUP IMAGE
        img1 = image[j,]
        img2 = image[k,]
        imgs.append((1-a)*img1 + a*img2)
        
        # MAKE CUTMIX LABEL
        lab1 = label[j,]
        lab2 = label[k,]
        labs.append((1-a)*lab1 + a*lab2)
            
    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
    image2 = tf.reshape(tf.stack(imgs),(BATCH_SIZE,DIM,DIM,3))
    label2 = tf.reshape(tf.stack(labs),(BATCH_SIZE,nb_classes))
    return image2,label2

# %% [code]
# gridmask
def transform(image, inv_mat, image_shape):
    h, w, c = image_shape
    cx, cy = w//2, h//2
    new_xs = tf.repeat( tf.range(-cx, cx, 1), h)
    new_ys = tf.tile( tf.range(-cy, cy, 1), [w])
    new_zs = tf.ones([h*w], dtype=tf.int32)
    old_coords = tf.matmul(inv_mat, tf.cast(tf.stack([new_xs, new_ys, new_zs]), tf.float32))
    old_coords_x, old_coords_y = tf.round(old_coords[0, :] + w//2), tf.round(old_coords[1, :] + h//2)
    clip_mask_x = tf.logical_or(old_coords_x<0, old_coords_x>w-1)
    clip_mask_y = tf.logical_or(old_coords_y<0, old_coords_y>h-1)
    clip_mask = tf.logical_or(clip_mask_x, clip_mask_y)
    old_coords_x = tf.boolean_mask(old_coords_x, tf.logical_not(clip_mask))
    old_coords_y = tf.boolean_mask(old_coords_y, tf.logical_not(clip_mask))
    new_coords_x = tf.boolean_mask(new_xs+cx, tf.logical_not(clip_mask))
    new_coords_y = tf.boolean_mask(new_ys+cy, tf.logical_not(clip_mask))
    old_coords = tf.cast(tf.stack([old_coords_y, old_coords_x]), tf.int32)
    new_coords = tf.cast(tf.stack([new_coords_y, new_coords_x]), tf.int64)
    rotated_image_values = tf.gather_nd(image, tf.transpose(old_coords))
    rotated_image_channel = list()
    for i in range(c):
        vals = rotated_image_values[:,i]
        sparse_channel = tf.SparseTensor(tf.transpose(new_coords), vals, [h, w])
        rotated_image_channel.append(tf.sparse.to_dense(sparse_channel, default_value=0, validate_indices=False))
    return tf.transpose(tf.stack(rotated_image_channel), [1,2,0])

def random_rotate(image, angle, image_shape):
    def get_rotation_mat_inv(angle):
        # transform to radian
        angle = math.pi * angle / 180
        cos_val = tf.math.cos(angle)
        sin_val = tf.math.sin(angle)
        one = tf.constant([1], tf.float32)
        zero = tf.constant([0], tf.float32)
        rot_mat_inv = tf.concat([cos_val, sin_val, zero, -sin_val, cos_val, zero, zero, zero, one], axis=0)
        rot_mat_inv = tf.reshape(rot_mat_inv, [3,3])
        return rot_mat_inv
    angle = float(angle) * tf.random.normal([1],dtype='float32')
    rot_mat_inv = get_rotation_mat_inv(angle)
    return transform(image, rot_mat_inv, image_shape)


def GridMask(image_height, image_width, d1, d2, rotate_angle=1, ratio=0.5):
    h, w = image_height, image_width
    hh = int(np.ceil(np.sqrt(h*h+w*w)))
    hh = hh+1 if hh%2==1 else hh
    d = tf.random.uniform(shape=[], minval=d1, maxval=d2, dtype=tf.int32)
    l = tf.cast(tf.cast(d,tf.float32)*ratio+0.5, tf.int32)

    st_h = tf.random.uniform(shape=[], minval=0, maxval=d, dtype=tf.int32)
    st_w = tf.random.uniform(shape=[], minval=0, maxval=d, dtype=tf.int32)

    y_ranges = tf.range(-1 * d + st_h, -1 * d + st_h + l)
    x_ranges = tf.range(-1 * d + st_w, -1 * d + st_w + l)

    for i in range(0, hh//d+1):
        s1 = i * d + st_h
        s2 = i * d + st_w
        y_ranges = tf.concat([y_ranges, tf.range(s1,s1+l)], axis=0)
        x_ranges = tf.concat([x_ranges, tf.range(s2,s2+l)], axis=0)

    x_clip_mask = tf.logical_or(x_ranges < 0 , x_ranges > hh-1)
    y_clip_mask = tf.logical_or(y_ranges < 0 , y_ranges > hh-1)
    clip_mask = tf.logical_or(x_clip_mask, y_clip_mask)

    x_ranges = tf.boolean_mask(x_ranges, tf.logical_not(clip_mask))
    y_ranges = tf.boolean_mask(y_ranges, tf.logical_not(clip_mask))

    hh_ranges = tf.tile(tf.range(0,hh), [tf.cast(tf.reduce_sum(tf.ones_like(x_ranges)), tf.int32)])
    x_ranges = tf.repeat(x_ranges, hh)
    y_ranges = tf.repeat(y_ranges, hh)

    y_hh_indices = tf.transpose(tf.stack([y_ranges, hh_ranges]))
    x_hh_indices = tf.transpose(tf.stack([hh_ranges, x_ranges]))

    y_mask_sparse = tf.SparseTensor(tf.cast(y_hh_indices, tf.int64),  tf.zeros_like(y_ranges), [hh, hh])
    y_mask = tf.sparse.to_dense(y_mask_sparse, 1, False)

    x_mask_sparse = tf.SparseTensor(tf.cast(x_hh_indices, tf.int64), tf.zeros_like(x_ranges), [hh, hh])
    x_mask = tf.sparse.to_dense(x_mask_sparse, 1, False)

    mask = tf.expand_dims( tf.clip_by_value(x_mask + y_mask, 0, 1), axis=-1)

    mask = random_rotate(mask, rotate_angle, [hh, hh, 1])
    mask = tf.image.crop_to_bounding_box(mask, (hh-h)//2, (hh-w)//2, image_height, image_width)

    return mask

def apply_grid_mask(image, image_shape, PROBABILITY = gridmask_rate):
    AugParams = {
        'd1' : 100,
        'd2': 160,
        'rotate' : 45,
        'ratio' : 0.3
    }
    
        
    mask = GridMask(image_shape[0], image_shape[1], AugParams['d1'], AugParams['d2'], AugParams['rotate'], AugParams['ratio'])
    if image_shape[-1] == 3:
        mask = tf.concat([mask, mask, mask], axis=-1)
        mask = tf.cast(mask,tf.float32)
        #print(mask.shape) # (299,299,3)

# 会报错，放弃
#     imgs = []
#     BATCH_SIZE=len(image)
#     for j in range(BATCH_SIZE):
#         P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.int32)
#         if P==1:
#             imgs.append(image[j,]*mask)
#         else:
#             imgs.append(image[j,])
#     return tf.cast(imgs,tf.float32)

        
    # 整个batch启用或者不启用
    P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.int32)
    if P==1:
        return image*mask
    else:
        return image

def gridmask(img_batch, label_batch):
    return apply_grid_mask(img_batch, (img_size,img_size, 3)), label_batch

# %% [code]
def get_train_dataset(train_paths,train_labels=None):

    # num_parallel_calls并发处理数据的并发数
    train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels.astype(np.float32))).map(decode_image, num_parallel_calls=AUTO)

    #train_dataset = train_dataset.cache().map(data_aug, num_parallel_calls=AUTO).repeat()
    train_dataset = train_dataset.map(data_aug, num_parallel_calls=AUTO).repeat()
    
    if bool_rotation_transform:
        train_dataset =train_dataset.map(rotation_transform)
                     
    train_dataset = train_dataset.shuffle(512).batch(BATCH_SIZE,drop_remainder=True)


    if cutmix_rate:  
        print('启用cutmix')
        train_dataset =train_dataset.map(cutmix, num_parallel_calls=AUTO)
    if mixup_rate:  
        print('启用mixup')
        train_dataset =train_dataset.map(mixup, num_parallel_calls=AUTO)
    if gridmask_rate:
        print('启用gridmask')
        train_dataset =train_dataset.map(gridmask, num_parallel_calls=AUTO)
    if (cutmix_rate or mixup_rate):
        train_dataset =train_dataset.unbatch().shuffle(512).batch(BATCH_SIZE)


    # repeat()代表无限制复制原始数据，这里可以用count指明复制份数，但要注意要比fit中的epochs大才可
    # 直接调用repeat()的话，生成的序列就会无限重复下去
    # prefetch: prefetch next batch while training (autotune prefetch buffer size)
    train_dataset = train_dataset.prefetch(AUTO)

    return train_dataset

# %% [code]
# #暂时取消这一段，减少爆内存问题
# try:
#     view_train_dataset=get_train_dataset(train_paths,train_labels)
# except:
#     view_train_dataset=get_train_dataset(train_paths)
    
# it = view_train_dataset.__iter__()

# #看看train_dataset 是否正常显示
# show_x, show_y = it.next()#TPU下File system scheme '[local]' not implemented
# print(show_x.shape,'\n\n',show_y[0])

# plt.figure(figsize=(12, 6))
# for i in range(1):
#     plt.subplot(2, 4, i+1)
#     plt.imshow(show_x[i])

# %% [code]
def get_validation_dataset(valid_paths,valid_labels=None):
    dataset = tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
    dataset = dataset.map(decode_image, num_parallel_calls=AUTO)
    

    
    

    #dataset = dataset.cache()
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

# %% [code]
#生成测试集
def re_produce_test_dataset(test_paths):
    test_dataset = tf.data.Dataset.from_tensor_slices(test_paths)
    test_dataset = test_dataset.map(decode_image, num_parallel_calls=AUTO)
    

    

    if bool_tta:
        test_dataset = test_dataset.cache().map(data_aug, num_parallel_calls=AUTO)

    test_dataset = test_dataset.batch(BATCH_SIZE)
    return test_dataset

# %% [code]
# #tta时可重复运行这块观察是否多次运行时生成了不同的测试数据
# view_dataset = re_produce_test_dataset(test_paths[:8])
# it = view_dataset.__iter__()
# show_x= it.next()
# try:
#     print(show_x.shape)
# except:
#     print(show_x[0].shape)
#     show_x=show_x[0]
    
# plt.figure(figsize=(12, 6))
# for i in range(1):
#     plt.subplot(2, 4, i+1)
#     plt.imshow(show_x[i])

# %% [code]
if special_monitor=='auc':
    my_metrics.append(tf.keras.metrics.AUC(name='auc'))


#创建模型
def get_model():
    with strategy.scope():
        base_model =  efn.EfficientNetB7(weights=pre_trained, include_top=False, pooling='avg', input_shape=(img_size, img_size, 3))
        x = base_model.output
        predictions = Dense(nb_classes, activation=dense_activation)(x)
        model = Model(inputs=base_model.input, outputs=predictions)

    if label_smoothing_rate:
        print('启用label_smoothing')
        my_loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing_rate)
    elif bool_focal_loss:

        my_loss = tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.AUTO)

    else:
        my_loss='categorical_crossentropy'

    model.compile(optimizer=tf.keras.optimizers.Adam(lr_if_without_scheduler), 
                  loss=my_loss,
                  metrics=my_metrics
                 )

    return model

# %% [code]
callbacks=[]
if bool_lr_scheduler:
    callbacks.append(lr_callback)

# %% [code]
def training():
    probabilities =[]
    
    if cross_validation_folds:
        
        histories = []

        # early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 3)
        kfold = KFold(cross_validation_folds , shuffle = True)

        i=1




        for trn_ind, val_ind in kfold.split(train_paths,train_labels):
            print(); print('#'*25)
            print('### FOLD',i)
            print('#'*25)
            
            # print(trn_ind)
            print(val_ind)

            # 暂停checkpoint，防止爆内存
            # 每轮都应该重置 ModelCheckpoint
            # ch_p1 = ModelCheckpoint(filepath="temp_best.h5", monitor='val_accuracy', save_weights_only=True,verbose=1,save_best_only=True)


            if special_monitor=='auc': 
                ch_p1 = ModelCheckpoint(filepath="temp_best.h5", monitor='val_auc', mode='max',save_weights_only=True,verbose=1,save_best_only=True)

            temp_callbacks=callbacks.copy()
            temp_callbacks.append(ch_p1)

            trn_paths = np.array(train_paths)[trn_ind]
            val_paths=np.array(train_paths)[val_ind]
            
            

            trn_labels = train_labels[trn_ind]
            val_labels=train_labels[val_ind]
            test_paths = val_paths

            model = get_model()
            history = model.fit(
                get_train_dataset(trn_paths,trn_labels), 
                steps_per_epoch = trn_labels.shape[0]//BATCH_SIZE,

            

                epochs = EPOCHS,
                callbacks = temp_callbacks,
                validation_data = (get_validation_dataset(val_paths,val_labels)),
            )
            
            

            i+=1
            histories.append(history)

            #用val_loss最小的权重来预测
            model.load_weights("temp_best.h5")
            prob = model.predict(re_produce_test_dataset(test_paths), verbose=1)       
            
            
            probabilities.append(prob)
            probabilities.append(val_labels)
            break
            
            

    #if not cross_validation_folds:
    else:

        model = get_model()



        histories = model.fit(
            get_train_dataset(train_paths,train_labels), 
            steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,




            callbacks=callbacks,
            epochs=EPOCHS
        )
        
        
    return model,histories,probabilities

# %% [code]
#画history咯
def display_training_curves(training, title, subplot, validation=None):
    if subplot%10==1: # set up the subplots on the first call
        plt.subplots(figsize=(10,10), facecolor='#F0F0F0')
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    if validation is not None:
        ax.plot(validation)
    ax.set_title('model '+ title)
    ax.set_ylabel(title)
    #ax.set_ylim(0.28,1.05)
    ax.set_xlabel('epoch')
    if validation is not None:
        ax.legend(['train', 'valid.'])
    else:
        ax.legend(['train'])

# %% [code]


# %% [code]
def draw_training_curves(histories):
    # 写开，防止交叉验证时out of memory导致什么都没保存
    if cross_validation_folds:


    #然后画画- -
        for h in range(len(histories)):
            display_training_curves(histories[h].history['loss'], 'loss', 211, histories[h].history['val_loss'])
            display_training_curves(histories[h].history['accuracy'], 'accuracy', 212, histories[h].history['val_accuracy'])

    #if not cross_validation_folds:
    else:
        display_training_curves(histories.history['loss'], 'loss', 211)
        display_training_curves(histories.history['accuracy'], 'accuracy', 212)

# %% [code]
def predict(model,probabilities):
    if cross_validation_folds:
        y_pred = np.mean(probabilities,axis =0)
    #if not cross_validation_folds:
    else:
        if bool_tta:
            probabilities = []
            for i in range(tta_times+1):
                print('TTA Number: ',i,'\n')
                test_dataset = re_produce_test_dataset(test_paths)
                probabilities.append(model.predict(test_dataset))
                y_pred = np.mean(probabilities,axis =0)


        else:
            test_dataset = re_produce_test_dataset(test_paths)       
            y_pred = model.predict(test_dataset)
            
    return y_pred

# %% [code]
model,histories,probabilities=training()
draw_training_curves(histories)
y_pred=predict(model,probabilities)

# %% [code]
%matplotlib inline



from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, savename, title='Confusion Matrix',classes = paths):

    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')
    
    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    
    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()

# %% [code]
# val_labels = [train_labels[i] for i in [7,9,13,15,38,40,56,57,58,61,67,70,72,74,76,88,92,95
# ,100,107,111,116,124,126,130,134,138,139,140,148,149,150,151,154,155,160
# ,164,167,171,172,193,197,201,207,209,216,226,232,234,238,241,244,245,249
# ,250,254,259,262,264,269,270,273,275,284,300,306,307,308,313,318,320,321
# ,322,334,337,339,344,349,354,356,359,364,371,379,387,391,394,413,418,423
# ,424,426,459,462,464,465,466,468,470,471,485,487,502,507,508,513,516,518
# ,529,536,544,547,557,566,567,568,570,572,584,589,599,604,605,610,611,614
# ,615,619,621,632,637,638,639,640,654,669,671,675,682,684,685,692,693,694
# ,702,707,710,727,734,741,743,762,767,770,774,778,781,788]]
# val_labels = np.array(val_labels)
# test_dataset = re_produce_test_dataset([train_paths[i] for i in [7,9,13,15,38,40,56,57,58,61,67,70,72,74,76,88,92,95
# ,100,107,111,116,124,126,130,134,138,139,140,148,149,150,151,154,155,160
# ,164,167,171,172,193,197,201,207,209,216,226,232,234,238,241,244,245,249
# ,250,254,259,262,264,269,270,273,275,284,300,306,307,308,313,318,320,321
# ,322,334,337,339,344,349,354,356,359,364,371,379,387,391,394,413,418,423
# ,424,426,459,462,464,465,466,468,470,471,485,487,502,507,508,513,516,518
# ,529,536,544,547,557,566,567,568,570,572,584,589,599,604,605,610,611,614
# ,615,619,621,632,637,638,639,640,654,669,671,675,682,684,685,692,693,694
# ,702,707,710,727,734,741,743,762,767,770,774,778,781,788]])

# %% [code]
# pred = model.predict(test_dataset)

# %% [code]
pred = (np.array([np.argmax(ele) for ele in probabilities[0]])).astype(int)
val_labels = (np.array([np.argmax(ele) for ele in probabilities[1]])).astype(int)

# %% [code]
print(pred)

# %% [code]
cm = confusion_matrix((val_labels), np.array(pred))
plot_confusion_matrix(cm, 'confusion_matrix.png', title='confusion matrix')

# %% [code]
#############画图部分
fpr, tpr, threshold = metrics.roc_curve(val_labels, pred)
roc_auc = metrics.auc(fpr, tpr)
plt.figure(figsize=(6,6))
plt.title('Validation ROC')
plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

# %% [code]
def sub(name):
    ret=[]
    for i in y_pred:
        if i[0]>=i[1]:
            ret.append('AD')
        elif i[0]<i[1]:
            ret.append('CN')

    sub = pd.read_csv('../input/petpet/pet.csv')
    sub.loc[:, 'label'] = ret

    print(len(sub))

    sub.to_csv(name, index=False)
    sub.head(30)

# %% [code]
sub('submission.csv')

# %% [code]
#保留副本，防止错误操作
train_paths_copy=train_paths.copy()
train_labels_copy=train_labels.copy()

# %% [code]
if bool_pseudo:
    print("启用伪标签")
    
    train_paths=train_paths_copy.copy()
    train_labels=train_labels_copy.copy()
    
    
    print(len(train_paths),len(train_labels))
    for i in range(len(y_pred)):
        print(y_pred[i])


        if y_pred[i][0]>=0.9:
            train_paths=np.append(train_paths,test_paths[i])
            train_labels=np.append(train_labels,[[1,0]],axis = 0)
        elif y_pred[i][0]<=0.1:
            train_paths=np.append(train_paths,test_paths[i])
            train_labels=np.append(train_labels,[[0,1]],axis = 0)
    print(len(train_paths),len(train_labels))
    
else:
    for i in range(len(y_pred)):
        print(y_pred[i])

# %% [code]
if bool_pseudo:
    model,histories,probabilities=training()
    draw_training_curves(histories)
    y_pred=predict(model,probabilities)
    sub('submission_pseudo.csv')
