import io
import sys
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import tensorlayer as tl
from tensorflow.python.framework import ops
import ConvLSTMCell as clstm

def c3d_clstm(inputs, num_classes, reuse, is_training):
  """Builds the Conv3D-ConvLSTM Networks."""
  with tf.device('/gpu:0'):
    with tf.variable_scope('Conv3D_ConvLSTM', reuse=reuse):
      tl.layers.set_name_reuse(reuse)
      if inputs.get_shape().ndims!=5:
        raise Exception("The input dimension of 3DCNN must be rank 5")
      network_input = tl.layers.InputLayer(inputs, name='input_layer')      #Input Layer
      # 3DCNN-BN Layer 1
      conv3d_1 = tl.layers.Conv3dLayer(network_input,
                                        act=tf.identity,
                                        shape=[3,3,3,3,64],
                                        strides=[1,1,1,1,1],
                                        padding='SAME',
                                        name='Conv3d_1')
      conv3d_1 = tl.layers.BatchNormLayer(layer=conv3d_1, 
                                        act=tf.nn.relu,
                                        is_train=is_training,
                                        name='BatchNorm_1')
      pool3d_1 = tl.layers.PoolLayer(conv3d_1,
                                        ksize=[1,1,2,2,1],
                                        strides=[1,1,2,2,1],
                                        padding='SAME',
                                        pool = tf.nn.max_pool3d,
                                        name='Pool3D_1')
      # 3DCNN-Incep Layer 2
      conv3d_2_3x3 = tl.layers.Conv3dLayer(pool3d_1, 
                                        act=tf.identity, 
                                        shape=[3,3,3,64,128], 
                                        strides=[1,1,1,1,1],
                                        padding='SAME',
                                        name='Conv3d_2_3x3')
      conv3d_2_3x3 = tl.layers.BatchNormLayer(layer=conv3d_2_3x3, 
                                        act=tf.nn.relu,
                                        is_train=is_training, 
                                        name='BatchNorm_2_3x3')
      pool3d_2 = tl.layers.PoolLayer(conv3d_2_3x3,
                                        ksize=[1,2,2,2,1],
                                        strides=[1,2,2,2,1],
                                        padding='SAME',
                                        pool = tf.nn.max_pool3d,
                                        name='Pool3D_2')
      # 3DCNN-Resnet Layer 1
      conv3d_3a_3x3 = tl.layers.Conv3dLayer(pool3d_2, 
                                        act=tf.identity, 
                                        shape=[3,3,3,128,256],
                                        strides=[1,1,1,1,1],
                                        padding='SAME',
                                        name='Conv3d_3a_3x3')
      conv3d_3b_3x3 = tl.layers.Conv3dLayer(conv3d_3a_3x3, 
                                        act=tf.identity, 
                                        shape=[3,3,3,256,256],
                                        strides=[1,1,1,1,1],
                                        padding='SAME',
                                        name='Conv3d_3b_3x3')
      conv3d_3_3x3 = tl.layers.BatchNormLayer(layer=conv3d_3b_3x3, 
                                        act=tf.nn.relu,
                                        is_train=is_training, 
                                        name='BatchNorm_3_3x3')
#      pool3d_3 = tl.layers.PoolLayer(conv3d_3_3x3,
#                                        ksize=[1,2,2,2,1],
#                                        strides=[1,2,2,2,1],
#                                        padding='SAME',
#                                        pool = tf.nn.max_pool3d,
#                                        name='Pool3D_3')
      # ConvLstm Layer
      shape3d = conv3d_3_3x3.outputs.get_shape().as_list()
      num_steps = shape3d[1]
      convlstm1 = tl.layers.RNNLayer(conv3d_3_3x3,
                                        cell_fn=clstm.ConvLSTMCell,
                                        cell_init_args={'state_is_tuple':False},
                                        n_hidden=256,
                                        initializer=tf.random_uniform_initializer(-0.1, 0.1),
                                        n_steps=num_steps,
                                        return_last=False,
                                        return_seq_2d=False,
                                        name='clstm_layer_1')
      convlstm2 = tl.layers.RNNLayer(convlstm1,
                                        cell_fn=clstm.ConvLSTMCell,
                                        cell_init_args={'state_is_tuple':False},
                                        n_hidden=384,
                                        initializer=tf.random_uniform_initializer(-0.1, 0.1),
                                        n_steps=num_steps,
                                        return_last=True,
                                        return_seq_2d=False,
                                        name='clstm_layer_2')
      # SPP Layer 1
      spp_bin_1 = tl.layers.PoolLayer(convlstm2,
                                        ksize=[1,28,28,1],
                                        strides=[1,28,28,1],
                                        padding='SAME',
                                        pool = tf.nn.max_pool,
                                        name='SPP_1')
      spp_bin_1 = tl.layers.FlattenLayer(spp_bin_1, 
                                        name='Flatten_SPP_1')
      spp_bin_2 = tl.layers.PoolLayer(convlstm2,
                                        ksize=[1,14,14,1],
                                        strides=[1,14,14,1],
                                        padding='SAME',
                                        pool = tf.nn.max_pool,
                                        name='SPP_2')
      spp_bin_2 = tl.layers.FlattenLayer(spp_bin_2, 
                                        name='Flatten_SPP_2')
      spp_bin_4 = tl.layers.PoolLayer(convlstm2,
                                        ksize=[1,7,7,1],
                                        strides=[1,7,7,1],
                                        padding='SAME',
                                        pool = tf.nn.max_pool,
                                        name='SPP_4')
      spp_bin_4 = tl.layers.FlattenLayer(spp_bin_4, 
                                        name='Flatten_SPP_4')
      spp_bin_7 = tl.layers.PoolLayer(convlstm2,
                                        ksize=[1,4,4,1],
                                        strides=[1,4,4,1],
                                        padding='SAME',
                                        pool = tf.nn.max_pool,
                                        name='SPP_8')
      spp_bin_7 = tl.layers.FlattenLayer(spp_bin_7, 
                                        name='Flatten_SPP_7')
      concat_spp = tl.layers.ConcatLayer(layer=[spp_bin_1,
                                              spp_bin_2,
                                              spp_bin_4,
                                              spp_bin_7],
                                        concat_dim=1,
                                        name='Concat_SPP')
      # FC Layer 1
      classes = tl.layers.DropconnectDenseLayer(concat_spp, 
                                        keep=0.5,
                                        n_units=num_classes,
                                        act=tf.identity,
                                        name='Classes')
    return classes

