#!/usr/bin/python

"""配置参数文件"""
class Config:
    def __init__(self):
        self.obs_dim = 3*3
        self.act_dim = 3*1
        self.actionBound = [[0.1,3],[0.1,3],[0.1,3]]

        self.MAX_EPISODE = 60
        self.MAX_STEP = 500
        self.batch_size = 256

        self.update_every = 50
        self.noise = 3

        self.if_load_weights = False