#!/usr/bin/python

"""配置参数文件"""
class Config:
    def __init__(self):
        # self.obs_dim = 4*10 # [q_x, q_y, v_x, v_y]*10
        # self.act_dim = 4*1 # 逃跑、追击、支援、搜索

        self.MAX_EPISODE = 60
        self.MAX_STEP = 500
        self.batch_size = 64         # 每次从缓冲区采样的批量大小

        # self.update_every = 50
        # self.noise = 3

        # self.if_load_weights = False

        self.minimal_size = 500
        self.buffer_size = 10000
        self.hidden_dim = 128        # 隐藏层维度
        self.epsilon = 0.01          # 探索率
        self.gamma = 0.98            # 折扣因子
        self.lr = 2e-3               # 学习率，控制神经网络参数更新的步长
        self.target_update = 10      # 目标网络更新频率（每训练多少次更新一次）