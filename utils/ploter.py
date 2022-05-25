# -*- coding:utf-8 -*-
import os
import pickle

import matplotlib.pyplot as plt


class Ploter(object):
    def __init__(self, performance_file):
        self.performance_file = performance_file
        self.performance = pickle.load(file=open(self.performance_file, "rb"))
        self.epoch_index = []
        self.success_rate = []
        self.average_reward = []
        self.average_wrong_disease = []
        self.__prepare_data()

    def __prepare_data(self, epoch_size=50):
        epoch_size = max(epoch_size, len(self.performance.keys()))
        for epoch_index in range(0, epoch_size, 1):
            self.epoch_index.append(epoch_index)
            self.success_rate.append(self.performance[epoch_index]["success_rate"])
            self.average_reward.append(self.performance[epoch_index]["average_reward"])
            self.average_wrong_disease.append(self.performance[epoch_index]["average_wrong_disease"])

    def plot(self):
        size = len(self.epoch_index)
        plt.plot(self.epoch_index[0:size], self.success_rate[0:size], label="DQN Agent", linewidth=1)
        plt.xlabel("Simulation Epoch")
        plt.ylabel("Success Rate")
        plt.title("Learning Curve")
        plt.hlines(0.95, 0, size, label="Rule Agent", linewidth=1, colors="purple")
        # plt.hlines(0.06, 0, size, label="Random Agent", linewidth=1, colors="r")
        plt.grid(True)
        # plt.legend(loc="lower right")
        plt.legend(loc='center right')
        # plt.savefig(save_name,dpi=400)

        plt.show()


if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(os.getcwd()), "model/dqn/learning_rate")
    file_name = "learning_rate_d_e_agent_T8_lr0.001_RFS50_RFF-30_RFNCY-10_RFIRS30_gamma1.0_epsilon0.1_RIDNone_499.p"
    save_path = os.path.join(os.path.dirname(os.getcwd()), "model/plot")
    save_name = file_name[:-2] + ".png"
    save_name = os.path.join(save_path, save_name)
    file_name = os.path.join(file_path, file_name)

    ploter = Ploter(file_name)
    ploter.plot()
    plt.savefig(save_name)
    print("save success")
