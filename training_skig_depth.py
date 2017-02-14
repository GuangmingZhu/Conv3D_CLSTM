import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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
import cStringIO

seq_len = 32
batch_size = 13
n_epoch = 165
learning_rate = 0.01
decay_steps = 10000
decay_rate  = 0.1
weight_decay= 0.00004
print_freq = 20
queue_num = 5
start_step = 0

num_classes = 10
dataset_name = 'skig_depth_s1'
training_datalist = '/ssd/gmzhu/SKIG/trte_splits/training_depth_list1.txt'
testing_datalist = '/ssd/gmzhu/SKIG/trte_splits/testing_depth_list1.txt'
model_prefix='/raid/gmzhu/tensorflow/c3d_clstm_net2'

curtime = '%s' % datetime.now()
d = curtime.split(' ')[0]
t = curtime.split(' ')[1]
strtime = '%s%s%s-%s%s%s' %(d.split('-')[0],d.split('-')[1],d.split('-')[2], 
                            t.split(':')[0],t.split(':')[1],t.split(':')[2])

saved_stdout = sys.stdout
mem_log = cStringIO.StringIO()
sys.stdout = mem_log
logfile = './log/training_%s_%s.log' %(dataset_name, strtime)
log = open(logfile, 'w')

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [batch_size, seq_len, 112, 112, 3], name='x')
y = tf.placeholder(tf.int32, shape=[batch_size, ], name='y')
  
networks = net.c3d_clstm(x, num_classes, False, True)
networks_y = networks.outputs
networks_y_op = tf.argmax(tf.nn.softmax(networks_y), 1)
networks_cost = tl.cost.cross_entropy(networks_y, y)
correct_pred = tf.equal(tf.cast(networks_y_op, tf.int32), y)
networks_accu = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
predictons = net.c3d_clstm(x, num_classes, True, False)
predicton_y_op = tf.argmax(tf.nn.softmax(predictons.outputs),1)
predicton_accu = tf.reduce_mean(tf.cast(tf.equal(tf.cast(predicton_y_op, tf.int32), y), tf.float32))
  
l2_cost = tf.contrib.layers.l2_regularizer(weight_decay)(networks.all_params[0]) + \
          tf.contrib.layers.l2_regularizer(weight_decay)(networks.all_params[6]) + \
          tf.contrib.layers.l2_regularizer(weight_decay)(networks.all_params[12]) + \
          tf.contrib.layers.l2_regularizer(weight_decay)(networks.all_params[14]) + \
          tf.contrib.layers.l2_regularizer(weight_decay)(networks.all_params[20]) + \
          tf.contrib.layers.l2_regularizer(weight_decay)(networks.all_params[22]) + \
          tf.contrib.layers.l2_regularizer(weight_decay)(networks.all_params[24]) 
cost = networks_cost + l2_cost 
  
# Decay the learning rate exponentially based on the number of steps.
global_step = tf.Variable(start_step*2, trainable=False)
lr1 = tf.train.exponential_decay(learning_rate,
                                global_step,
                                decay_steps,
                                decay_rate,
                                staircase=True)
lr2 = tf.train.exponential_decay(learning_rate*10,
                                global_step,
                                decay_steps,
                                decay_rate,
                                staircase=True)
var_list1 = networks.all_params[:24] 
var_list2 = networks.all_params[24:]
opt1 = tf.train.GradientDescentOptimizer(lr1)
opt2 = tf.train.GradientDescentOptimizer(lr2)
grads = tf.gradients(cost, var_list1 + var_list2)
grads1 = grads[:len(var_list1)]
grads2 = grads[len(var_list1):]
train_op1 = opt1.apply_gradients(zip(grads1, var_list1), global_step=global_step)
train_op2 = opt2.apply_gradients(zip(grads2, var_list2), global_step=global_step)
train_op = tf.group(train_op1, train_op2)

sess.run(tf.initialize_all_variables())

if start_step>0:
  load_params = tl.files.load_npz(name='%s_model_iter_%d.npz'%(dataset_name, start_step))
  tl.files.assign_params(sess, load_params, networks)
else:
  load_params = tl.files.load_npz(name='isogr_rgb_model_iter_60000.npz')
  tl.files.assign_params(sess, load_params[0:24], networks)
networks.print_params(True)
  
# Data Reading
X_train,y_train = data.load_video_list(training_datalist)
X_tridx = np.asarray(np.arange(0, len(y_train)), dtype=np.int32)
y_train = np.asarray(y_train, dtype=np.int32)
X_test,y_test = data.load_video_list(testing_datalist)
X_teidx = np.asarray(np.arange(0, len(y_test)), dtype=np.int32)
y_test  = np.asarray(y_test, dtype=np.int32)
  
X_data_a  = np.empty((batch_size*queue_num, seq_len, 112, 112, 3),float)      
y_label_a = np.empty((batch_size*queue_num,),int)

full_flg  = np.zeros((queue_num, 1))
rdwr_lock = threading.Lock()

def training_data_read():
  wr_pos = 0
  for i in range(n_epoch):
    for X_indices, y_labels in tl.iterate.minibatches(X_tridx,
                                                      y_train, 
                                                      batch_size, 
                                                      shuffle=True):
      # 1. Waiting
      while True:
        rdwr_lock.acquire()
        if full_flg[wr_pos] == 1:
          rdwr_lock.release()
          time.sleep(1)
          continue
        rdwr_lock.release()
        break
      # 2. Reading data
      image_path = []
      image_fcnt = []
      image_olen = []
      is_training = []
      for data_a in range(batch_size):
        X_index_a = X_indices[data_a]
        key_str = '%06d' % X_index_a
        image_path.append(X_train[key_str]['videopath'])
        image_fcnt.append(X_train[key_str]['framecnt'])
        image_olen.append(seq_len)
        is_training.append(True) # Training
      image_info = zip(image_path,image_fcnt,image_olen,is_training)
      X_data_a[wr_pos*batch_size:(wr_pos+1)*batch_size,:,:,:,:] = \
                  tl.prepro.threading_data([_ for _ in image_info], 
                                           data.prepare_skig_depth_data)
      y_label_a[wr_pos*batch_size:(wr_pos+1)*batch_size] = y_labels
      # 3. Update flags
      rdwr_lock.acquire()
      full_flg[wr_pos] = 1
      rdwr_lock.release()
      wr_pos = (wr_pos+1)%queue_num      
  
wr_thread = threading.Thread(target=training_data_read)
wr_thread.start()

# Output the saved logs to stdout and the opened log file
sys.stdout = saved_stdout
mem_log.seek(0)
print mem_log.read()
mem_log.seek(0)
log.writelines(['%s' % mem_log.read()])
log.flush()
mem_log.close()

step = start_step
rd_pos = 0
for epoch in range(n_epoch):
  # Train Stage
  for _,_ in tl.iterate.minibatches(X_tridx, 
                                    y_train, 
                                    batch_size, 
                                    shuffle=True):
    # 1. Read data for each batch      
    while True:
      rdwr_lock.acquire()
      if full_flg[rd_pos] == 0:
        rdwr_lock.release()
        time.sleep(1)
        continue
      rdwr_lock.release()
      break
    # 2. Training
    feed_dict = {x: X_data_a[rd_pos*batch_size:(rd_pos+1)*batch_size,:,:,:,:], 
                 y: y_label_a[rd_pos*batch_size:(rd_pos+1)*batch_size]}
    feed_dict.update(networks.all_drop)
    start_time = time.time()
    _,loss_value,lr_value,acc = sess.run([train_op,cost,lr1,networks_accu], feed_dict=feed_dict)
    duration = time.time() - start_time
    # 3. Update flags
    rdwr_lock.acquire()
    full_flg[rd_pos] = 0
    rdwr_lock.release()
    rd_pos = (rd_pos+1)%queue_num
    # 4. Statistics
    if step%print_freq == 0:
      average_acc = acc
      total_loss = loss_value
      training_time = duration
    else:
      average_acc += acc
      total_loss = total_loss + loss_value
      training_time = training_time + duration
    if (step+1)%print_freq == 0:
      training_bps = batch_size*print_freq / training_time
      average_loss = total_loss / print_freq
      average_acc = average_acc / print_freq
      format_str = ('%s: iter = %d, lr=%f, average_loss = %.2f average_acc = %.6f (training: %.1f batches/sec)')
      print (format_str % (datetime.now(), step+1, lr_value, average_loss, average_acc, training_bps))
      log.writelines([format_str % (datetime.now(), step+1, lr_value, average_loss, average_acc, training_bps), 
                      '\n'])
      log.flush()
    step = step + 1
    if step%300 == 0:
      tl.files.save_npz(networks.all_params, 
                        name='%s_model_iter_%d.npz'%(dataset_name, step), 
                        sess=sess)
      print("Model saved in file: %s_model_iter_%d.npz" %(dataset_name, step))
    
      # Test Stage
      average_accuracy = 0.0
      test_iterations = 0;
      for X_indices, y_label_t in tl.iterate.minibatches(X_teidx, 
                                                         y_test, 
                                                         batch_size, 
                                                         shuffle=True):
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
                                            data.prepare_skig_depth_data)
        feed_dict = {x: X_data_t, y: y_label_t}
        dp_dict = tl.utils.dict_to_one(predictons.all_drop)
        feed_dict.update(dp_dict)
        _,accu_value = sess.run([predicton_y_op, predicton_accu], feed_dict=feed_dict)
        average_accuracy = average_accuracy + accu_value
        test_iterations = test_iterations + 1
      average_accuracy = average_accuracy / test_iterations
      format_str = ('%s: epoch = %d, average_accuracy = %.6f')
      print (format_str % (datetime.now(), epoch, average_accuracy))
      log.writelines([format_str % (datetime.now(), epoch, average_accuracy), '\n'])
      log.flush()

# In the end, close TensorFlow session.
log.close()
sess.close()
