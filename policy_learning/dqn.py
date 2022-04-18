# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf


class DQN1(object):
    """
    One layer.tensorflow.
    """

    def __init__(self, input_size, hidden_size, output_size, parameter):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # self.checkpoint_path = parameter.get("checkpoint_path")
        # self.log_dir = parameter.get("log_dir")
        self.parameter = parameter
        self.learning_rate = parameter.get("dqn_learning_rate")
        self.update_target_network_operations = []
        self.__build_model()

    def __build_model(self):
        device = self.parameter.get("device_for_tf")
        # with tf.device("/device:GPU:1"):
        # with tf.device("/gpu:0"):
        with tf.device("/device:GPU:0"):
            self.graph = tf.Graph()

            with self.graph.as_default():
                self.input = tf.placeholder(dtype=tf.float64, shape=(None, self.input_size), name="input")
                self.target_value = tf.placeholder(dtype=tf.float64, shape=(None, self.output_size),
                                                   name="target_value")
                # Target network
                self.regularizer = tf.contrib.layers.l2_regularizer(
                    scale=0.1)  # tf.contrib.layers.l2_regularizer：返回一个执行L2正则化的函数，该函数可用于对权重应用L2正则化。较小的L2值有助于防止训练数据过度拟合。scale: 正则项的系数

                with tf.variable_scope(name_or_scope="target_network"):
                    self.target_network_variables = {}
                    self.target_output = self._build_layer(variables_dict=self.target_network_variables,
                                                           input=self.input,
                                                           input_size=self.input_size,
                                                           output_size=self.output_size,
                                                           weights_key="w1",
                                                           bias_key="b1")

                # Current network.
                with tf.variable_scope(name_or_scope="current_network"):
                    self.current_network_variables = {}
                    self.current_output = self._build_layer(variables_dict=self.current_network_variables,
                                                            input=self.input,
                                                            input_size=self.input_size,
                                                            output_size=self.output_size,
                                                            weights_key="w1",
                                                            bias_key="b1")

                    # Regularization.
                    for key, value in self.current_network_variables.items():
                        if "w" in key: tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                                            value)  # tf.add_to_collection(‘list_name’, element)：将元素element添加到列表list_name中
                # Updating target network.
                with tf.name_scope(name="ops_of_updating_target_network"):
                    self.update_target_network_operations = self._update_target_network_operations()

                # Optimization.
                reg_variables = tf.get_collection(
                    tf.GraphKeys.REGULARIZATION_LOSSES)  # tf.get_collection：该函数的作用是从一个collection中取出全部变量，形成一个列表，key参数中输入的是collection的名称
                self.reg_loss = tf.contrib.layers.apply_regularization(self.regularizer,
                                                                       reg_variables)  # 应用正则化方法到参数上；regularizer:前面创建的正则化方法；reg_variables：执行正则化方法的参数列表
                self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.target_value - self.current_output), axis=1),
                                           name="loss") \
                            + self.reg_loss  # 在损失函数上加上正则项是防止过拟合的一个重要方法

                tf.summary.scalar("loss", self.loss)  # 用来显示标量信息；一般在画loss,accuary时会用到这个函数。
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(
                    loss=self.loss)
                self.initializer = tf.global_variables_initializer()
                self.model_saver = tf.train.Saver()  # TensorFlow通过tf.train.Saver类实现神经网络模型的保存和提取。
            self.graph.finalize()  # 结束这个图，使这个图变成只读。

        # Visualizing graph.
        # self.summary_writer = tf.summary.FileWriter(logdir=self.log_dir + "train", graph=self.graph)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        self.session = tf.Session(graph=self.graph, config=config)
        self.session.run(self.initializer)
        """
        if self.parameter.get("train_mode") != 1:
            self.restore_model(self.parameter.get("saved_model"))
        """

    def singleBatch(self, batch, params):
        # state, agent_action, reward, next_state, episode_over
        gamma = params.get('gamma', 0.9)
        Xs = []
        next_Xs = []
        for i, x in enumerate(batch):
            state_pre = x[0]
            next_state_pre = x[3]
            Xs.append(state_pre)
            next_Xs.append(next_state_pre)
        next_Ys = self._predict_target(Xs=next_Xs, params=params)[0]
        Ys_label = self.predict(Xs=Xs, params=params)[0]  # numpy.ndarray

        for i, x in enumerate(batch):
            reward = x[2]  # int
            episode_over = x[4]  # bool
            action = x[1]  # int
            next_max_y = np.max(next_Ys[i])
            target_y = reward
            if not episode_over:
                target_y = float(gamma * next_max_y + reward)
            Ys_label[i][action] = target_y

        feed_dict = {self.input: Xs, self.target_value: Ys_label}
        loss = self.session.run(self.loss, feed_dict=feed_dict)  # For return
        self.session.run(self.optimizer, feed_dict=feed_dict)
        return {"loss": loss}

    def predict(self, Xs, **kwargs):
        feed_dict = {self.input: Xs}
        Ys = self.session.run(self.current_output, feed_dict=feed_dict)
        max_index = np.argmax(Ys, axis=1)  # numpy.argmax(array, axis) 用于返回一个numpy数组中最大值的索引值。
        return Ys, max_index[0]

    def _predict_target(self, Xs, params, **kwargs):
        feed_dict = {self.input: Xs}
        Ys = self.session.run(self.target_output, feed_dict=feed_dict)
        max_index = np.argmax(Ys, axis=1)
        return Ys, max_index[0]

    def save_model(self, model_performance, episodes_index, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path
        # agent_id = self.parameter.get("agent_id")
        # dqn_id = self.parameter.get("dqn_id")
        # service_number = self.parameter.get("service_number")
        success_rate = model_performance["success_rate"]
        average_reward = model_performance["average_reward"]
        average_turn = model_performance["average_turn"]
        average_wrong_service = model_performance["average_wrong_service"]
        model_file_name = "model_d" + "_agent" + "_dqn" + "_s" + str(success_rate) + "_r" + str(
            average_reward) + "_t" + str(average_turn) + "_wd" + str(
            average_wrong_service) + "_e" + str(episodes_index)
        save_path = checkpoint_path + model_file_name
        self.model_saver.save(sess=self.session, save_path=save_path)
        print('model saved')

    def restore_model(self, saved_model):
        print("loading trained model")
        self.model_saver.restore(sess=self.session, save_path=saved_model)

    def update_target_network(self):
        self.session.run(fetches=self.update_target_network_operations)

    def _update_target_network_operations(self):
        update_target_network_operations = []
        for key in self.current_network_variables.keys():
            update = tf.assign(ref=self.target_network_variables[key],
                               value=self.current_network_variables[key].value())
            update_target_network_operations.append(update)
        return update_target_network_operations

    def _build_layer(self, variables_dict, input, input_size, output_size, weights_key, bias_key):
        with self.graph.as_default():
            weights = tf.get_variable(name=weights_key, shape=(input_size, output_size), dtype=tf.float64)
            bias = tf.get_variable(name=bias_key, shape=(output_size), dtype=tf.float64)
            variables_dict[weights_key] = weights
            variables_dict[bias_key] = bias
            tf.summary.scalar(name=weights_key, tensor=weights)
            tf.summary.scalar(name=bias_key, tensor=bias)
            output = tf.nn.relu(tf.add(tf.matmul(input, weights), bias), name="output")
        return output
