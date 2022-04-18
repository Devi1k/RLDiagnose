# import tensorflow as tf
# import os
# import numpy as np
#
# a = tf.Variable(1., tf.float32)
# b = tf.Variable(2., tf.float32)
# num = 10
#
# model_save_path = './model/'
# model_name = 'model'
#
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     init_op = tf.global_variables_initializer()
#     sess.run(init_op)
#     for step in np.arange(num):
#         c = sess.run(tf.add(a, b))
#         # checkpoint_path = os.path.join(model_save_path, model_name)
#         # 默认最多同时存放 5 个模型
#         saver.save(sess, os.path.join(model_save_path, model_name), global_step=step)

import tensorflow as tf

a = tf.Variable(1., tf.float32)
b = tf.Variable(2., tf.float32)
num = 10

model_save_path = './model/'
model_name = 'model'

saver = tf.train.Saver()

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    ckpt = tf.train.get_checkpoint_state(model_save_path)
    # 载入模型，不需要提供模型的名字，会通过 checkpoint 文件定位到最新保存的模型
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    print("load success")


