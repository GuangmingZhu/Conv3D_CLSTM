import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import io
import sys
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import tensorlayer as tl
import inputs as data
import c3d_clstm as net 
import time
from datetime import datetime
import threading

seq_len = 32
batch_size = 12

num_classes = 249
dataset_name = 'isogr'
model_prefix='/raid/gmzhu/tensorflow/c3d_clstm_net2'

curtime = '%s' % datetime.now()
d = curtime.split(' ')[0]
t = curtime.split(' ')[1]
strtime = '%s%s%s-%s%s%s' %(d.split('-')[0],d.split('-')[1],d.split('-')[2], 
                            t.split(':')[0],t.split(':')[1],t.split(':')[2])

x = tf.placeholder(tf.float32, [batch_size, seq_len, 112, 112, 3], name='x')
y = tf.placeholder(tf.int32, shape=[batch_size, ], name='y')
  
sess = tf.InteractiveSession()

networks = net.c3d_clstm(x, num_classes, False, False)
network_pred = tf.nn.softmax(networks.outputs)
network_y_op = tf.argmax(tf.nn.softmax(networks.outputs),1)
network_accu = tf.reduce_mean(tf.cast(tf.equal(tf.cast(network_y_op, tf.int32), y), tf.float32))
  
sess.run(tf.initialize_all_variables())

# RGB
testing_datalist = '/ssd/dataset/IsoGD_Image/valid_rgb_list.txt'
X_test,y_test = data.load_video_list(testing_datalist)
X_teidx = np.asarray(np.arange(0, len(y_test)), dtype=np.int32)
y_test  = np.asarray(y_test, dtype=np.int32)
rgb_prediction = np.zeros((len(y_test),num_classes), dtype=np.float32)
load_params = tl.files.load_npz(name='%s/train_results_paper/isogr_rgb_model_strategy_3.npz'%(model_prefix))
tl.files.assign_params(sess, load_params, networks)
#networks.print_params(True)
average_accuracy = 0.0
test_iterations = 0
print '%s: rgb testing' % datetime.now()
for X_indices, y_label_t in tl.iterate.minibatches(X_teidx, 
                                                   y_test, 
                                                   batch_size, 
                                                   shuffle=False):
  # Read data for each batch      
  image_path = []
  image_fcnt = []
  image_olen = []
  is_training = []
  for data_a in range(batch_size):
    X_index_a = X_indices[data_a]
    key_str = '%06d' % X_index_a
    image_path.append(X_test[key_str]['videopath'])
    image_fcnt.append(X_test[key_str]['framecnt'])
    image_olen.append(seq_len)
    is_training.append(False) # Testing
    image_info = zip(image_path,image_fcnt,image_olen,is_training)
  X_data_t = tl.prepro.threading_data([_ for _ in image_info], 
                                      data.prepare_isogr_rgb_data)
  feed_dict = {x: X_data_t, y: y_label_t}
  dp_dict = tl.utils.dict_to_one(networks.all_drop)
  feed_dict.update(dp_dict)
  predict_value,accu_value = sess.run([network_pred, network_accu], feed_dict=feed_dict)
  rgb_prediction[test_iterations*batch_size:(test_iterations+1)*batch_size,:]=predict_value
  average_accuracy = average_accuracy + accu_value
  test_iterations = test_iterations + 1
average_accuracy = average_accuracy / test_iterations
format_str = ('%s: rgb average_accuracy = %.6f')
print (format_str % (datetime.now(), average_accuracy))

# Depth
testing_datalist = '/ssd/dataset/IsoGD_Image/valid_depth_list.txt'
X_test,y_test = data.load_video_list(testing_datalist)
X_teidx = np.asarray(np.arange(0, len(y_test)), dtype=np.int32)
y_test  = np.asarray(y_test, dtype=np.int32)
depth_prediction = np.zeros((len(y_test),num_classes), dtype=np.float32)
load_params = tl.files.load_npz(name='%s/train_results_paper/isogr_depth_model_strategy_3.npz'%(model_prefix))
tl.files.assign_params(sess, load_params, networks)
#networks.print_params(True)
average_accuracy = 0.0
test_iterations = 0
print '%s: depth testing' % datetime.now()
for X_indices, y_label_t in tl.iterate.minibatches(X_teidx, 
                                                   y_test, 
                                                   batch_size, 
                                                   shuffle=False):
  # Read data for each batch      
  image_path = []
  image_fcnt = []
  image_olen = []
  is_training = []
  for data_a in range(batch_size):
    X_index_a = X_indices[data_a]
    key_str = '%06d' % X_index_a
    image_path.append(X_test[key_str]['videopath'])
    image_fcnt.append(X_test[key_str]['framecnt'])
    image_olen.append(seq_len)
    is_training.append(False) # Testing
    image_info = zip(image_path,image_fcnt,image_olen,is_training)
  X_data_t = tl.prepro.threading_data([_ for _ in image_info], 
                                      data.prepare_isogr_depth_data)
  feed_dict = {x: X_data_t, y: y_label_t}
  dp_dict = tl.utils.dict_to_one(networks.all_drop)
  feed_dict.update(dp_dict)
  predict_value,accu_value = sess.run([network_pred, network_accu], feed_dict=feed_dict)
  depth_prediction[test_iterations*batch_size:(test_iterations+1)*batch_size,:]=predict_value
  average_accuracy = average_accuracy + accu_value
  test_iterations = test_iterations + 1
average_accuracy = average_accuracy / test_iterations
format_str = ('%s: depth average_accuracy = %.6f')
print (format_str % (datetime.now(), average_accuracy))

fusion_prediction = rgb_prediction + depth_prediction
prediction_values = tf.argmax(fusion_prediction, 1)
final_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(prediction_values, tf.int32), y_test), tf.float32))
print final_accuracy.eval()

# In the end, close TensorFlow session.
sess.close()
