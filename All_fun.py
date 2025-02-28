#!/usr/bin/python

import numpy as np
import random
import heapq
from find_enemy_area import FindEnemyArea
import math
from find_safe_point import FindSafeSpot
import torch
import cv2
import matplotlib.pyplot as plt

class IIFDS:
    """使用IIFDS类训练时每次必须reset"""

    def __init__(self):
        """基本参数："""
        self.V1 = 0.4  # 速度大小的最大值限制
        self.V2 = 0.4
        self.threshold = 1  # 最大打击距离阈值，在该打击距离下，无人车无法隔墙打击
        self.threshold2 = 0.4  # 搜索任务的到达距离阈值
        self.threshold3 = 0.4  # 逃跑任务的到达距离阈值
        self.stepSize = 0.1  # 时间间隔步长
        self.lam = 0.4  # 避障参数，越大考虑障碍物速度越明显
        self.numberofuav = 10  # 无人车数量
        self.uavR = 0.3  # 无人车半径
        self.num_com = 1  # 路径规划时考虑的邻居数量
        self.obsR = 0.3  # 障碍物半径
        self.R_1 = 5  # 针对敌军的感知半径
        self.R_2 = 5  # 针对友军的感知半径
        self.missle_num = 100  # 最大子弹填充数量
        self.hit_angle = np.pi / 2  # 子弹攻击角度
        self.hit_rate = 0.5  # 子弹命中概率
        self.HP_num = 1  # 初始生命值
        self.end_predict = 0  # 开始预测的回合
        self.vel_fill_missle = 0.5  # 子弹填充速度
        # 初始位置设置
        self.start1 = np.array([random.uniform(0 - 10.85, 1.5 - 10.85), random.uniform(2 - 7.45, 7 - 7.45), 0.8])
        self.start11 = np.array([random.uniform(0 - 10.85, 1.5 - 10.85), random.uniform(7 - 7.45, 12.9 - 7.45), 0.8])
        self.start13 = np.array([random.uniform(1.5 - 10.85, 6 - 10.85), random.uniform(2 - 7.45, 2.5 - 7.45), 0.8])
        self.start14 = np.array([random.uniform(2.5 - 10.85, 5.4 - 10.85), random.uniform(3.6 - 7.45, 5.3 - 7.45), 0.8])
        self.start5 = np.array([random.uniform(1.9 - 10.85, 6 - 10.85), random.uniform(6.6 - 7.45, 9 - 7.45), 0.8])
        self.start16 = np.array([random.uniform(2.4 - 10.85, 5.5 - 10.85), random.uniform(10 - 7.45, 12 - 7.45), 0.8])
        self.start7 = np.array(
            [random.uniform(20.2 - 10.85, 21.7 - 10.85), random.uniform(9.2 - 7.45, 12.9 - 7.45), 0.8])

        self.start18 = np.array([random.uniform(6.5 - 10.85, 7.9 - 10.85), random.uniform(2 - 7.45, 7 - 7.45), 0.8])
        self.start9 = np.array([random.uniform(7.9 - 10.85, 9.3 - 10.85), random.uniform(2 - 7.45, 7 - 7.45), 0.8])
        self.start10 = np.array([random.uniform(6.5 - 10.85, 7.9 - 10.85), random.uniform(7 - 7.45, 12.9 - 7.45), 0.8])
        self.start12 = np.array([random.uniform(7.9 - 10.85, 9.3 - 10.85), random.uniform(7 - 7.45, 12.9 - 7.45), 0.8])

        self.start2 = np.array([random.uniform(9.7 - 10.85, 13.6 - 10.85), random.uniform(2 - 7.45, 3.5 - 7.45), 0.8])
        self.start20 = np.array(
            [random.uniform(11.4 - 10.85, 13.6 - 10.85), random.uniform(6.5 - 7.45, 8.2 - 7.45), 0.8])
        self.start4 = np.array(
            [random.uniform(9.7 - 10.85, 13.6 - 10.85), random.uniform(11.5 - 7.45, 12.9 - 7.45), 0.8])

        self.start15 = np.array([random.uniform(14.1 - 10.85, 15.2 - 10.85), random.uniform(2 - 7.45, 7 - 7.45), 0.8])
        self.start6 = np.array([random.uniform(14.1 - 10.85, 15.2 - 10.85), random.uniform(7 - 7.45, 12.9 - 7.45), 0.8])

        self.start17 = np.array(
            [random.uniform(16.2 - 10.85, 19.2 - 10.85), random.uniform(2.5 - 7.45, 5.9 - 7.45), 0.8])
        self.start8 = np.array(
            [random.uniform(16.2 - 10.85, 19.2 - 10.85), random.uniform(8.9 - 7.45, 12.5 - 7.45), 0.8])

        self.start19 = np.array([random.uniform(20.2 - 10.85, 21.7 - 10.85), random.uniform(2 - 7.45, 5.6 - 7.45), 0.8])
        self.start3 = np.array(
            [random.uniform(20.2 - 10.85, 21.7 - 10.85), random.uniform(5.6 - 7.45, 9.2 - 7.45), 0.8])

        self.start_b = [self.start1, self.start6, self.start3, self.start4, self.start5]
        self.start_r = [self.start11, self.start15, self.start13, self.start19, self.start17]
        for i in range(int(self.numberofuav / 2)):
            self.start_b[i][2] = self.start_b[i][2] * 2 / 3
            self.start_r[i][2] = self.start_r[i][2] * 2 / 3

        self.ass = [5, 6, 7, 8, 9, 0, 1, 2, 3, 4]

        self.safePos = []

        self.start = []
        for i in range(int(self.numberofuav / 2)):
            self.start.append(self.start_b[i])
        for i in range(int(self.numberofuav / 2)):
            self.start.append(self.start_r[i])
        self.goal = [self.start[self.ass[0]], self.start[self.ass[1]], self.start[self.ass[2]], self.start[self.ass[3]],
                     self.start[self.ass[4]], self.start[self.ass[5]], self.start[self.ass[6]], self.start[self.ass[7]],
                     self.start[self.ass[8]], self.start[self.ass[9]]]

        self.timelog = 0  # 时间，用来计算动态障碍的位置
        self.timeStep = 0.1

        self.xmax = 10 / 180 * np.pi  # 偏航角速度最大值  每个步长允许变化的角度
        self.gammax = 10 / 180 * np.pi  # 爬升角速度最大值  每个步长允许变化的角度
        self.maximumClimbingAngle = 100 / 180 * np.pi  # 最大爬升角
        self.maximumSubductionAngle = - 75 / 180 * np.pi  # 最大俯冲角

        self.vObs = None
        self.vObsNext = None

    def detect(self, uavPos, flag_uav, ta_index, HP_index, obsCenter):
        all_opp = []
        all_nei_c2e = []
        all_nei = []
        all_close_opp = []
        all_close_nei = []
        for i in range(self.numberofuav):
            distance1 = np.ones([1, int(self.numberofuav)]) * np.inf
            distance2 = np.ones([1, int(self.numberofuav)]) * np.inf
            opp = []
            nei = []
            nei_c2e = []
            if i < int(self.numberofuav / 2):
                for j in range(self.numberofuav):
                    if j != i and j >= int(self.numberofuav / 2):  # 敌方判断
                        if flag_uav[j] == 0:  # 存活判断
                            if self.is_within_perception_range(self.R_1, self.R_1, uavPos[i][0:2], uavPos[j][0:2]):  # 感知判断
                                flag_undetected = 0
                                for k in range(len(obsCenter)):
                                    if len(self.line_intersect_circle((obsCenter[k][0], obsCenter[k][1], self.obsR),
                                                                      (uavPos[i][0], uavPos[i][1]),
                                                                      (uavPos[j][0], uavPos[j][1]))) == 2:
                                        flag_undetected = 1  # 隔墙无法感知敌军
                                        break
                                if flag_undetected == 0:
                                    opp.append(j)  # 存放能感知到的敌军
                                    distance1[0][j] = self.distanceCost(uavPos[i], uavPos[j]) / self.R_1 + HP_index[
                                        j] / self.HP_num  # 敌军信息包括位置、血量
                    elif j != i and j < int(self.numberofuav / 2):  # 友方判断
                        if flag_uav[j] == 0:  # 存活判断
                            if self.is_within_perception_range(self.R_2, self.R_2, uavPos[i][0:2], uavPos[j][0:2]):  # 记录感知半径范围内的友军
                                flag_undetected = 0
                                for k in range(len(obsCenter)):
                                    if len(self.line_intersect_circle((obsCenter[k][0], obsCenter[k][1], self.obsR),
                                                                      (uavPos[i][0], uavPos[i][1]),
                                                                      (uavPos[j][0], uavPos[j][1]))) == 2:
                                        flag_undetected = 1  # 隔墙无法感知友军
                                        break
                                if flag_undetected == 0:
                                    nei.append(j)  # 存放能感知到的友军
                                    if ta_index[-1][j] == 0 or ta_index[-1][j] == -2:  # 记录追击和逃跑的友军
                                        if ta_index[-1][j] == -2:
                                            distance2[0][j] = self.distanceCost(uavPos[i], uavPos[j]) / self.R_2 + \
                                                              HP_index[j]/ self.HP_num + self.HP_num   # 友军信息包括位置、血量
                                        else:
                                            distance2[0][j] = self.distanceCost(uavPos[i], uavPos[j]) / self.R_2 + \
                                                              HP_index[j]/ self.HP_num
                                        nei_c2e.append(j)
                uav_catch = heapq.nsmallest(1, distance1[0])
                index1 = list(map(distance1[0].tolist().index, uav_catch))
                uav_contact = heapq.nsmallest(1, distance2[0])
                index2 = list(map(distance2[0].tolist().index, uav_contact))
                all_opp.append(opp)
                all_nei.append(nei)
                all_nei_c2e.append(nei_c2e)
                all_close_opp.append(index1[0])
                all_close_nei.append(index2[0])
            else:  # 与上述基本相同，但存放敌军和友军信息时，只考虑位置
                for j in range(self.numberofuav):
                    if j != i and j < int(self.numberofuav / 2):
                        if flag_uav[j] == 0:
                            if self.is_within_perception_range(self.R_1, self.R_1, uavPos[i][0:2], uavPos[j][0:2]):
                                flag_undetected = 0
                                for k in range(len(obsCenter)):
                                    if len(self.line_intersect_circle((obsCenter[k][0], obsCenter[k][1], self.obsR),
                                                                      (uavPos[i][0], uavPos[i][1]),
                                                                      (uavPos[j][0], uavPos[j][1]))) == 2:
                                        flag_undetected = 1
                                        break
                                if flag_undetected == 0:
                                    opp.append(j)
                                    distance1[0][j] = self.distanceCost(uavPos[i], uavPos[j])
                    elif j != i and j >= int(self.numberofuav / 2):
                        if flag_uav[j] == 0:
                            if self.is_within_perception_range(self.R_2, self.R_2, uavPos[i][0:2], uavPos[j][0:2]):  # 记录感知半径范围内的友军
                                flag_undetected = 0
                                for k in range(len(obsCenter)):
                                    if len(self.line_intersect_circle((obsCenter[k][0], obsCenter[k][1], self.obsR),
                                                                      (uavPos[i][0], uavPos[i][1]),
                                                                      (uavPos[j][0], uavPos[j][1]))) == 2:
                                        flag_undetected = 1  # 隔墙无法感知友军
                                        break
                                if flag_undetected == 0:
                                    nei.append(j)
                                    if ta_index[-1][j] == 0 or ta_index[-1][j] == -2:  # 只记录追击和逃跑的友军
                                        if ta_index[-1][j] == -2:
                                            distance2[0][j] = self.distanceCost(uavPos[i], uavPos[j])
                                        else:
                                            distance2[0][j] = self.distanceCost(uavPos[i], uavPos[j])
                                        nei_c2e.append(j)
                uav_catch = heapq.nsmallest(1, distance1[0])
                index1 = list(map(distance1[0].tolist().index, uav_catch))
                uav_contact = heapq.nsmallest(1, distance2[0])
                index2 = list(map(distance2[0].tolist().index, uav_contact))
                all_opp.append(opp)
                all_nei.append(nei)
                all_nei_c2e.append(nei_c2e)
                all_close_opp.append(index1[0])
                all_close_nei.append(index2[0])
        return all_opp, all_nei, all_nei_c2e, all_close_opp, all_close_nei

    def Meanfield(self, uavPos, uavVel, all_opp, all_nei, all_nei_c2e):
        ave_opp_pos, ave_opp_vel, ave_nei_pos, ave_nei_vel, num_opp, num_nei_c2e = [], [], [], [], [], []
        for i in range(int(self.numberofuav)):
            num_opp.append(len(all_opp[i]))
            num_nei_c2e.append(len(all_nei_c2e[i]))
            ave_nei_pos.append((sum(uavPos[index] for index in all_nei[i]) + uavPos[i]) / (len(all_nei[i]) + 1))
            ave_nei_vel.append((sum(uavVel[index] for index in all_nei[i]) + uavVel[i]) / (len(all_nei[i]) + 1))
            if(len(all_opp[i]) == 0):
                ave_opp_pos.append(float('-inf'))
                ave_opp_vel.append(float('-inf'))
            else:
                ave_opp_pos.append(sum(uavPos[index] for index in all_opp[i]) / len(all_opp[i]))
                ave_opp_vel.append(sum(uavVel[index] for index in all_opp[i]) / len(all_opp[i]))
        return ave_opp_pos, ave_opp_vel, ave_nei_pos, ave_nei_vel, num_opp, num_nei_c2e
    
    def stateSelectionBlue(self, ave_opp_pos, ave_opp_vel, ave_nei_pos, ave_nei_vel, num_opp, num_nei_c2e, missle_index):
        task_index_blue = []
        for i in range(int(self.numberofuav / 2)):
            if missle_index[i] == 0:  # 只要弹药为空，就逃
                task_index_blue.append(-2)  # 逃逸
            else:
                if num_opp[i] != 0: # 发现敌方
                    if self.cos_cal(ave_nei_vel[i], ave_opp_pos[i] - ave_nei_pos[i]) >= self.cos_cal(ave_opp_vel[i],
                                                                                                -ave_opp_pos[i] + ave_nei_pos[i]):
                        task_index_blue.append(0)  # 追击
                    else:
                        task_index_blue.append(-2)  # 逃逸
                else:
                    if num_nei_c2e[i] != 0:  # 存在逃跑或追击的友军
                        task_index_blue.append(-1)  # 支援
                    else:
                        task_index_blue.append(-3)  # 搜索
                        
        return task_index_blue
    
    def stateSelectionRed(self, uavPos, uavVel, missle_index, all_opp, all_nei_c2e, all_close_opp):
        task_index_red = []
        for i in range(int(self.numberofuav / 2)):
            i = i + int(self.numberofuav / 2)
            if missle_index[i] == 0:  # 只要弹药为空，就逃
                task_index_red.append(-2)  # 逃逸
            else:
                if len(all_opp[i]) != 0: # 发现敌方
                    if self.cos_cal(uavVel[i], uavPos[all_close_opp[i]] - uavPos[i]) >= self.cos_cal(
                                uavVel[all_close_opp[i]], -uavPos[all_close_opp[i]] + uavPos[i]):
                        task_index_red.append(0)  # 追击
                    else:
                        task_index_red.append(-2)  # 逃逸
                else:
                    if len(all_nei_c2e[i]) != 0:  # 存在逃跑或追击的友军
                        task_index_red.append(-1)  # 支援
                    else:
                        task_index_red.append(-3)  # 搜索

        return task_index_red
    
    def allocation(self, uavPos,  goal,  pos_b, pos_r, ta_index,
               obsCenter,  all_close_opp, all_close_nei, task_index, epi):
        ass_index = []

        for i in range(self.numberofuav):
            if task_index[i] == -3: # 搜索
                if epi > self.end_predict and (
                                ta_index[-1][i] != -3 or epi % 5 == 0 or self.distanceCost(goal[i],
                                                                                           uavPos[
                                                                                               i]) < self.threshold2):
                    if i < self.numberofuav / 2:
                        finder = FindEnemyArea(pos_r, obsCenter, self.timeStep, self.obsR+self.uavR)
                    else:
                        finder = FindEnemyArea(pos_b, obsCenter, self.timeStep, self.obsR+self.uavR)
                    temp = finder.predict_trajectory(10)
                    try:
                        goal[i][0:2] = finder.find_nearest_center(temp, uavPos[i][0:2])
                    except Exception as e:
                        # print('未找到覆盖点')
                        pass
                ass_index.append(-2)
            elif task_index[i] == -2: # 逃逸
                if (ta_index[-1][i] != -2) or (
                    self.distanceCost(goal[i], uavPos[i]) < self.threshold3):  # 若状态刚切换为逃逸或到达逃逸点，重新计算逃逸目标点
                    if i < self.numberofuav / 2:
                        finder = FindSafeSpot(pos_r, uavPos[i][0:2], obsCenter, self.timeStep, self.obsR+self.uavR)
                    else:
                        finder = FindSafeSpot(pos_b, uavPos[i][0:2], obsCenter, self.timeStep, self.obsR+self.uavR)
                    temp = finder.predict_trajectory(10)
                    goal[i][0:2] = finder.find_safe_spot(list(temp.values()))
                ass_index.append(-1)
            elif task_index[i] == -1: # 支援
                goal[i] = uavPos[all_close_nei[i]]
                ass_index.append(all_close_nei[i])
            else: # 追击
                goal[i] = uavPos[all_close_opp[i]]
                ass_index.append(all_close_opp[i])

        return goal, ass_index

    def assign(self, uavPos, uavVel, goal, missle_index, epi, pos_b, pos_r, ta_index,
               obsCenter, all_opp, all_nei, all_nei_c2e, all_close_opp, all_close_nei):
        ass_index = []
        task_index = []

        for i in range(self.numberofuav):
            if missle_index[i] == 0:  # 只要弹药为空，就逃
                if (ta_index[-1][i] != -2) or (
                        self.distanceCost(goal[i], uavPos[i]) < self.threshold3):  # 若状态刚切换为逃逸或到达逃逸点，重新计算逃逸目标点
                    if i < self.numberofuav / 2:
                        finder = FindSafeSpot(pos_r, uavPos[i][0:2], obsCenter, self.timeStep, self.obsR+self.uavR)
                    else:
                        finder = FindSafeSpot(pos_b, uavPos[i][0:2], obsCenter, self.timeStep, self.obsR+self.uavR)
                    temp = finder.predict_trajectory(10)
                    goal[i][0:2] = finder.find_safe_spot(list(temp.values()))
                ass_index.append(-1)
                task_index.append(-2)  # 逃逸
            else:
                if len(all_opp[i]) != 0:
                    if i < self.numberofuav / 2:
                        ave_opp_pos = sum(uavPos[index] for index in all_opp[i]) / len(all_opp[i])
                        ave_opp_vel = sum(uavVel[index] for index in all_opp[i]) / len(all_opp[i])
                        ave_nei_pos = (sum(uavPos[index] for index in all_nei[i]) + uavPos[i]) / (len(all_nei[i]) + 1)
                        ave_nei_vel = (sum(uavVel[index] for index in all_nei[i]) + uavVel[i]) / (len(all_nei[i]) + 1)
                        if self.cos_cal(ave_nei_vel, ave_opp_pos - ave_nei_pos) >= self.cos_cal(ave_opp_vel,
                                                                                                -ave_opp_pos + ave_nei_pos):
                            goal[i] = uavPos[all_close_opp[i]]
                            ass_index.append(all_close_opp[i])
                            task_index.append(0)  # 追击
                        else:
                            if (ta_index[-1][i] != -2) or (
                                    self.distanceCost(goal[i], uavPos[i]) < self.threshold3):  # 若状态刚切换为逃逸或到达逃逸点
                                # ===========================
                                # 无人机敌情侦察模块（王莉）
                                # ===========================
                                finder = FindSafeSpot(pos_r, uavPos[i][0:2], obsCenter, self.timeStep, self.obsR+self.uavR)
                                temp = finder.predict_trajectory(10)
                                goal[i][0:2] = finder.find_safe_spot(list(temp.values()))
                                # ===========================
                            ass_index.append(-1)
                            task_index.append(-2)  # 逃逸
                    else:
                        if self.cos_cal(uavVel[i], uavPos[all_close_opp[i]] - uavPos[i]) >= self.cos_cal(
                                uavVel[all_close_opp[i]], -uavPos[all_close_opp[i]] + uavPos[i]):
                            goal[i] = uavPos[all_close_opp[i]]
                            ass_index.append(all_close_opp[i])
                            task_index.append(0)  # 追击
                        else:
                            if (ta_index[-1][i] != -2) or (
                                    self.distanceCost(goal[i], uavPos[i]) < self.threshold3):  # 若状态刚切换为逃逸或到达逃逸点
                                # ===========================
                                # 无人机敌情侦察模块（王莉）
                                # ===========================
                                finder = FindSafeSpot(pos_b, uavPos[i][0:2], obsCenter, self.timeStep, self.obsR+self.uavR)
                                temp = finder.predict_trajectory(10)
                                goal[i][0:2] = finder.find_safe_spot(list(temp.values()))
                                # ===========================
                            ass_index.append(-1)
                            task_index.append(-2)  # 逃逸
                else:
                    if len(all_nei_c2e[i]) != 0:  # 存在逃跑或追击的友军
                        goal[i] = uavPos[all_close_nei[i]]
                        ass_index.append(all_close_nei[i])
                        task_index.append(-1)  # 支援
                    else:
                        if epi > self.end_predict and (
                                ta_index[-1][i] != -3 or epi % 5 == 0 or self.distanceCost(goal[i],
                                                                                           uavPos[
                                                                                               i]) < self.threshold2):
                            if i < self.numberofuav / 2:
                                # ===========================
                                # 无人机敌情侦察模块（王莉）
                                # ===========================
                                finder = FindEnemyArea(pos_r, obsCenter, self.timeStep, self.obsR+self.uavR)
                                temp = finder.predict_trajectory(10)
                                try:
                                    goal[i][0:2] = finder.find_nearest_center(temp, uavPos[i][0:2])
                                except Exception as e:
                                    pass
                            else:
                                finder = FindEnemyArea(pos_b, obsCenter, self.timeStep, self.obsR+self.uavR)
                                temp = finder.predict_trajectory(10)
                                try:
                                    goal[i][0:2] = finder.find_nearest_center(temp, uavPos[i][0:2])
                                except Exception as e:
                                    pass
                                # ===========================
                        ass_index.append(-2)
                        task_index.append(-3)  # 搜索
        return goal, ass_index, task_index

    def cos_cal(self, a, b):
        return a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def calDynamicState(self, uavPos, uavVel, obsPos, obsVel, obs_num, goal, flag_uav):
        """强化学习模型获得的state。"""
        dic = {'uav1': [], 'uav2': [], 'uav3': [], 'uav4': [], 'uav5': [], 'uav6': [], 'uav7': [], 'uav8': [],
               'uav9': [], 'uav10': []}
        s = []
        for i in range(self.numberofuav):
            s1 = goal[i] - uavPos[i]
            s.append(s1)
        # 不仅考虑到观测障碍物 额外还能观测邻居无人车或障碍物
        distance = np.ones([self.numberofuav, int(self.numberofuav + obs_num)]) * np.inf
        for i in range(self.numberofuav):
            for j in range(self.numberofuav + obs_num):
                if j != i and j < int(self.numberofuav):
                    if flag_uav[j] == 0:
                        distance[i][j] = self.distanceCost(uavPos[i], uavPos[j])
                elif j != i and j >= self.numberofuav:
                    distance[i][j] = self.distanceCost(uavPos[i][0:2], obsPos[int(j - self.numberofuav)][0:2])

        z = []
        self.uav_com = np.zeros([self.numberofuav, self.num_com])
        self.index_com = np.zeros([self.numberofuav, self.num_com])
        for i in range(self.numberofuav):
            self.uav_com[i] = heapq.nsmallest(self.num_com, distance[i])
            self.index_com[i] = list(map(distance[i].tolist().index, self.uav_com[i]))
            for j in range(int(1)):
                if int(self.index_com[i][j]) < self.numberofuav:
                    z1 = (uavPos[int(self.index_com[i][j])] - uavPos[i]) * (
                            self.distanceCost(uavPos[int(self.index_com[i][j])],
                                              uavPos[i]) - 2 * self.uavR) / self.distanceCost(
                        uavPos[int(self.index_com[i][j])],
                        uavPos[i])
                    z.append(z1)
                    z2 = uavVel[int(self.index_com[i][j])]
                    z.append(z2)
                else:
                    z1 = (obsPos[int(self.index_com[i][j] - self.numberofuav)] - uavPos[i]) * (
                            self.distanceCost(obsPos[int(self.index_com[i][j] - self.numberofuav)][0:2],
                                              uavPos[i][0:2]) - (self.uavR + self.obsR)) / self.distanceCost(
                        obsPos[int(self.index_com[i][j] - self.numberofuav)][0:2],
                        uavPos[i][0:2])
                    z1[2] = 0
                    z.append(z1)
                    z2 = obsVel[int(self.index_com[i][j] - self.numberofuav)]
                    z2[2] = 0
                    z.append(z2)
        dic['uav1'].append(np.hstack((s[0], z[0], z[1])))
        dic['uav2'].append(np.hstack((s[1], z[2], z[3])))
        dic['uav3'].append(np.hstack((s[2], z[4], z[5])))
        dic['uav4'].append(np.hstack((s[3], z[6], z[7])))
        dic['uav5'].append(np.hstack((s[4], z[8], z[9])))
        dic['uav6'].append(np.hstack((s[5], z[10], z[11])))
        dic['uav7'].append(np.hstack((s[6], z[12], z[13])))
        dic['uav8'].append(np.hstack((s[7], z[14], z[15])))
        dic['uav9'].append(np.hstack((s[8], z[16], z[17])))
        dic['uav10'].append(np.hstack((s[9], z[18], z[19])))
        return dic

    def calRepulsiveMatrix(self, uavPos, obsCenter, cylinderR, row0, goal):
        n = self.partialDerivativeSphere(obsCenter, uavPos, cylinderR)
        tempD = self.distanceCost(uavPos, obsCenter) - cylinderR
        row = row0 * np.exp(1 - 1 / (self.distanceCost(uavPos, goal) * tempD))
        T = self.calculateT(obsCenter, uavPos, cylinderR)
        repulsiveMatrix = np.dot(-n, n.T) / T ** (1 / row) / np.dot(n.T, n)[0][0]
        return repulsiveMatrix

    def calTangentialMatrix(self, uavPos, obsCenter, cylinderR, theta, sigma0, goal):
        n = self.partialDerivativeSphere(obsCenter, uavPos, cylinderR)
        T = self.calculateT(obsCenter, uavPos, cylinderR)
        partialX = (uavPos[0] - obsCenter[0]) * 2 / cylinderR ** 2
        partialY = (uavPos[1] - obsCenter[1]) * 2 / cylinderR ** 2
        partialZ = (uavPos[2] - obsCenter[2]) * 2 / cylinderR ** 2
        tk1 = np.array([partialY, -partialX, 0], dtype=float).reshape(-1, 1)
        tk2 = np.array([partialX * partialZ, partialY * partialZ, -partialX ** 2 - partialY ** 2], dtype=float).reshape(
            -1, 1)
        originalPoint = np.array([np.cos(theta), np.sin(theta), 0]).reshape(1, -1)
        tk = self.trans(originalPoint, tk1.squeeze(), tk2.squeeze(), n.squeeze())
        tempD = self.distanceCost(uavPos, obsCenter) - cylinderR
        sigma = sigma0 * np.exp(1 - 1 / (self.distanceCost(uavPos, goal) * tempD))
        tangentialMatrix = tk.dot(n.T) / T ** (1 / sigma) / self.calVecLen(tk.squeeze()) / self.calVecLen(n.squeeze())
        return tangentialMatrix

    def calRepulsiveMatrix2(self, uavPos, obsCenter, cylinderR, row0, goal):
        n = self.partialDerivativeSphere2(obsCenter, uavPos, cylinderR)
        tempD = self.distanceCost(uavPos[0:2], obsCenter[0:2]) - cylinderR
        row = row0 * np.exp(1 - 1 / (self.distanceCost(uavPos, goal) * tempD))
        T = self.calculateT2(obsCenter, uavPos, cylinderR)
        repulsiveMatrix = np.dot(-n, n.T) / T ** (1 / row) / np.dot(n.T, n)[0][0]
        return repulsiveMatrix

    def calTangentialMatrix2(self, uavPos, obsCenter, cylinderR, theta, sigma0, goal):
        n = self.partialDerivativeSphere2(obsCenter, uavPos, cylinderR)
        T = self.calculateT2(obsCenter, uavPos, cylinderR)
        partialX = (uavPos[0] - obsCenter[0]) * 2 / cylinderR ** 2
        partialY = (uavPos[1] - obsCenter[1]) * 2 / cylinderR ** 2
        partialZ = 0
        tk1 = np.array([partialY, -partialX, 0], dtype=float).reshape(-1, 1)
        tk2 = np.array([partialX * partialZ, partialY * partialZ, -partialX ** 2 - partialY ** 2], dtype=float).reshape(
            -1, 1)
        originalPoint = np.array([np.cos(theta), np.sin(theta), 0]).reshape(1, -1)
        tk = self.trans(originalPoint, tk1.squeeze(), tk2.squeeze(), n.squeeze())
        tempD = self.distanceCost(uavPos[0:2], obsCenter[0:2]) - cylinderR
        sigma = sigma0 * np.exp(1 - 1 / (self.distanceCost(uavPos, goal) * tempD))
        tangentialMatrix = tk.dot(n.T) / T ** (1 / sigma) / self.calVecLen(tk.squeeze()) / self.calVecLen(n.squeeze())
        return tangentialMatrix

    def getvNext(self, q, v, obsq, obsv, qBefore_all, goal_all, flag_uav, arglist, actors_cur1, actors_cur2):

        obsDicq = self.calDynamicState(q, v, obsq, obsv, len(obsq), goal_all, flag_uav)  # 相对位置字典
        (obs_n_uav1, obs_n_uav2, obs_n_uav3, obs_n_uav4, obs_n_uav5, obs_n_uav6, obs_n_uav7, obs_n_uav8, obs_n_uav9,
         obs_n_uav10) = obsDicq['uav1'], obsDicq['uav2'], obsDicq['uav3'], obsDicq[
            'uav4'], obsDicq['uav5'], obsDicq['uav6'], obsDicq['uav7'], obsDicq[
                            'uav8'], obsDicq['uav9'], obsDicq['uav10']
        obs_n1 = obs_n_uav1 + obs_n_uav2 + obs_n_uav3 + obs_n_uav4 + obs_n_uav5
        obs_n2 = obs_n_uav6 + obs_n_uav7 + obs_n_uav8 + obs_n_uav9 + obs_n_uav10

        action_n1 = [agent(torch.from_numpy(obs).to(arglist.device, torch.float)).detach().cpu().numpy() \
                     for agent, obs in zip(actors_cur1, obs_n1)]
        action_n1 = np.clip(action_n1, arglist.action_limit_min, arglist.action_limit_max)
        action_n1 = action_n1.reshape(-1)

        action_n2 = [agent(torch.from_numpy(obs).to(arglist.device, torch.float)).detach().cpu().numpy() \
                     for agent, obs in zip(actors_cur2, obs_n2)]
        action_n2 = np.clip(action_n2, arglist.action_limit_min, arglist.action_limit_max)
        action_n2 = action_n2.reshape(-1)
        vNext = []
        for i in range(len(q)):
            uavPos = q[i]
            goal = goal_all[i]
            qBefore = qBefore_all[i]
            repulsiveMatrix = 0
            tangentialMatrix = 0
            ubar = 0

            if i < self.numberofuav / 2:
                row0 = action_n1[3 * i]
                sigma0 = action_n1[3 * i + 1]
                theta = action_n1[3 * i + 2]
                u = self.initField(uavPos, self.V1, goal)
            else:
                row0 = action_n2[3 * (i - int(self.numberofuav / 2))]
                sigma0 = action_n2[3 * (i - int(self.numberofuav / 2)) + 1]
                theta = action_n2[3 * (i - int(self.numberofuav / 2)) + 2]
                u = self.initField(uavPos, self.V2, goal)

            for j in range(int(self.num_com)):

                if int(self.index_com[i][j]) < self.numberofuav:
                    repulsiveMatrix += self.calRepulsiveMatrix(uavPos, q[int(self.index_com[i][j])], 2 * self.uavR,
                                                               row0,
                                                               goal)
                    tangentialMatrix += self.calTangentialMatrix(uavPos, q[int(self.index_com[i][j])], 2 * self.uavR,
                                                                 theta,
                                                                 sigma0, goal)
                    M = np.eye(3) + repulsiveMatrix + tangentialMatrix
                    T_ = self.calculateT(q[int(self.index_com[i][j])], uavPos, 2 * self.uavR)
                    vp = np.exp(-T_ / self.lam) * v[int(self.index_com[i][j])]
                elif int(self.index_com[i][j]) >= self.numberofuav:
                    repulsiveMatrix += self.calRepulsiveMatrix2(uavPos,
                                                                obsq[int(self.index_com[i][j] - self.numberofuav)],
                                                                self.uavR + self.obsR, row0, goal)
                    tangentialMatrix += self.calTangentialMatrix2(uavPos,
                                                                  obsq[int(self.index_com[i][j] - self.numberofuav)],
                                                                  self.uavR + self.obsR, theta,
                                                                  sigma0, goal)
                    M = np.eye(3) + repulsiveMatrix + tangentialMatrix

                    T_ = self.calculateT2(obsq[int(self.index_com[i][j] - self.numberofuav)], uavPos,
                                          self.uavR + self.obsR)
                    vp = np.exp(-T_ / self.lam) * obsv[int(self.index_com[i][j] - self.numberofuav)]
                ubar += (M.dot(u - vp.reshape(-1, 1)).T + vp.reshape(1, -1)).squeeze()

            # 限制ubar的模长，避免进入障碍内部后轨迹突变
            if self.calVecLen(ubar) > 5:
                ubar = ubar / self.calVecLen(ubar) * 5
            if qBefore[0] is None:
                uavNextPos = uavPos + ubar * self.stepSize
            else:
                uavNextPos = uavPos + ubar * self.stepSize
                _, _, _, _, uavNextPos = self.kinematicConstrant(uavPos, qBefore, uavNextPos)
            uavNextPos[2] = uavPos[2]

            for j in range(len(obsq)):
                if self.distanceCost(uavNextPos, obsq[j]) < (self.obsR + self.uavR):

                    point1 = [uavNextPos[0] + (uavNextPos[1] - uavPos[1]) * 1000,
                              uavNextPos[1] - (uavNextPos[0] - uavPos[0]) * 1000]
                    point2 = [uavNextPos[0] - (uavNextPos[1] - uavPos[1]) * 1000,
                              uavNextPos[1] + (uavNextPos[0] - uavPos[0]) * 1000]
                    cross_pos = self.line_intersect_circle((obsq[j][0], obsq[j][1], self.obsR + self.uavR + 0.1),
                                                           (point1[0], point1[1]), (point2[0], point2[1]))

                    try:
                        if (self.distanceCost(np.array([cross_pos[0][0], cross_pos[0][1]]),
                                              np.array([uavPos[0], uavPos[1]]))
                                < self.distanceCost(np.array([cross_pos[1][0], cross_pos[1][1]]),
                                                    np.array([uavPos[0], uavPos[1]]))):
                            uavNextPos[0] = cross_pos[0][0]
                            uavNextPos[1] = cross_pos[0][1]
                        else:
                            uavNextPos[0] = cross_pos[1][0]
                            uavNextPos[1] = cross_pos[1][1]
                    except Exception as e:
                        uavNextPos = obsq[j] + (self.obsR + self.uavR + 0.1) * (uavNextPos - obsq[j]) / np.linalg.norm(
                            uavNextPos - obsq[j])

            for j in range(len(q)):
                if j != i:
                    if self.distanceCost(uavNextPos, q[j]) < (2 * self.uavR):
                        point1 = [uavNextPos[0] + (uavNextPos[1] - uavPos[1]) * 1000,
                                  uavNextPos[1] - (uavNextPos[0] - uavPos[0]) * 1000]
                        point2 = [uavNextPos[0] - (uavNextPos[1] - uavPos[1]) * 1000,
                                  uavNextPos[1] + (uavNextPos[0] - uavPos[0]) * 1000]
                        cross_pos = self.line_intersect_circle((q[j][0], q[j][1], 2 * self.uavR + 0.2),
                                                               (point1[0], point1[1]), (point2[0], point2[1]))
                        try:
                            if (self.distanceCost(np.array([cross_pos[0][0], cross_pos[0][1]]),
                                                  np.array([uavPos[0], uavPos[1]]))
                                    < self.distanceCost(np.array([cross_pos[1][0], cross_pos[1][1]]),
                                                        np.array([uavPos[0], uavPos[1]]))):
                                uavNextPos[0] = cross_pos[0][0]
                                uavNextPos[1] = cross_pos[0][1]
                            else:
                                uavNextPos[0] = cross_pos[1][0]
                                uavNextPos[1] = cross_pos[1][1]
                        except Exception as e:
                            uavNextPos = q[j] + (2 * self.uavR + 0.1) * (uavNextPos - q[j]) / np.linalg.norm(
                                uavNextPos - q[j])

            if uavNextPos[0] < -11.5:
                uavNextPos[0] = -11.5 + 0.5
            if uavNextPos[0] > 11.5:
                uavNextPos[0] = 11.5 - 0.5
            if uavNextPos[1] < -6.5:
                uavNextPos[1] = -6.5 + 0.5
            if uavNextPos[1] > 6.5:
                uavNextPos[1] = 6.5 - 0.5
            # uavNextVel = np.array([0, 0, 0])
            # if self.cos_cal(uavNextPos[0:2] - uavPos[0:2], np.array([1, 0])) > np.sqrt(2) / 2:
            #     uavNextVel[0] = self.V1
            #     uavNextVel[1] = 0
            # elif self.cos_cal(uavNextPos[0:2] - uavPos[0:2], np.array([0, 1])) >= np.sqrt(2) / 2:
            #     uavNextVel[0] = 0
            #     uavNextVel[1] = self.V1
            # elif self.cos_cal(uavNextPos[0:2] - uavPos[0:2], np.array([-1, 0])) > np.sqrt(2) / 2:
            #     uavNextVel[0] = - self.V1
            #     uavNextVel[1] = 0
            # else:
            #     uavNextVel[0] = 0
            #     uavNextVel[1] = - self.V1
            uavNextVel = (uavNextPos - uavPos) / np.linalg.norm(uavNextPos - uavPos) * self.V1
            vNext.append(uavNextVel)
        return vNext

    def kinematicConstrant(self, q, qBefore, qNext):
        """
        运动学约束函数 返回(上一时刻航迹角，上一时刻爬升角，约束后航迹角，约束后爬升角，约束后下一位置qNext)
        """
        # 计算qBefore到q航迹角x1,gam1
        qBefore2q = q - qBefore
        if qBefore2q[0] != 0 or qBefore2q[1] != 0:
            x1 = np.arcsin(
                np.abs(qBefore2q[1] / np.sqrt(qBefore2q[0] ** 2 + qBefore2q[1] ** 2)))  # 这里计算的角限定在了第一象限的角 0-pi/2
            gam1 = np.arcsin(qBefore2q[2] / np.sqrt(np.sum(qBefore2q ** 2)))
        else:
            return None, None, None, None, qNext
        # 计算q到qNext航迹角x2,gam2
        q2qNext = qNext - q
        x2 = np.arcsin(np.abs(q2qNext[1] / np.sqrt(q2qNext[0] ** 2 + q2qNext[1] ** 2)))  # 这里同理计算第一象限的角度
        gam2 = np.arcsin(q2qNext[2] / np.sqrt(np.sum(q2qNext ** 2)))

        # 根据不同象限计算矢量相对于x正半轴的角度 0-2 * pi
        if qBefore2q[0] > 0 and qBefore2q[1] > 0:
            x1 = x1
        if qBefore2q[0] < 0 and qBefore2q[1] > 0:
            x1 = np.pi - x1
        if qBefore2q[0] < 0 and qBefore2q[1] < 0:
            x1 = np.pi + x1
        if qBefore2q[0] > 0 and qBefore2q[1] < 0:
            x1 = 2 * np.pi - x1
        if qBefore2q[0] > 0 and qBefore2q[1] == 0:
            x1 = 0
        if qBefore2q[0] == 0 and qBefore2q[1] > 0:
            x1 = np.pi / 2
        if qBefore2q[0] < 0 and qBefore2q[1] == 0:
            x1 = np.pi
        if qBefore2q[0] == 0 and qBefore2q[1] < 0:
            x1 = np.pi * 3 / 2

        # 根据不同象限计算与x正半轴的角度
        if q2qNext[0] > 0 and q2qNext[1] > 0:
            x2 = x2
        if q2qNext[0] < 0 and q2qNext[1] > 0:
            x2 = np.pi - x2
        if q2qNext[0] < 0 and q2qNext[1] < 0:
            x2 = np.pi + x2
        if q2qNext[0] > 0 and q2qNext[1] < 0:
            x2 = 2 * np.pi - x2
        if q2qNext[0] > 0 and q2qNext[1] == 0:
            x2 = 0
        if q2qNext[0] == 0 and q2qNext[1] > 0:
            x2 = np.pi / 2
        if q2qNext[0] < 0 and q2qNext[1] == 0:
            x2 = np.pi
        if q2qNext[0] == 0 and q2qNext[1] < 0:
            x2 = np.pi * 3 / 2

        # 约束航迹角x   xres为约束后的航迹角
        deltax1x2 = self.angleVec(q2qNext[0:2], qBefore2q[0:2])  # 利用点乘除以模长乘积求xoy平面投影的夹角
        if deltax1x2 < self.xmax:
            xres = x2
        elif x1 - x2 > 0 and x1 - x2 < np.pi:  # 注意这几个逻辑
            xres = x1 - self.xmax
        elif x1 - x2 > 0 and x1 - x2 > np.pi:
            xres = x1 + self.xmax
        elif x1 - x2 < 0 and x2 - x1 < np.pi:
            xres = x1 + self.xmax
        else:
            xres = x1 - self.xmax

        # 约束爬升角gam   注意：爬升角只用讨论在-pi/2到pi/2区间，这恰好与arcsin的值域相同。  gamres为约束后的爬升角
        if np.abs(gam1 - gam2) <= self.gammax:
            gamres = gam2
        elif gam2 > gam1:
            gamres = gam1 + self.gammax
        else:
            gamres = gam1 - self.gammax
        if gamres > self.maximumClimbingAngle:
            gamres = self.maximumClimbingAngle
        if gamres < self.maximumSubductionAngle:
            gamres = self.maximumSubductionAngle

        # 计算约束过后下一个点qNext的坐标
        Rq2qNext = self.distanceCost(q, qNext)
        deltax = Rq2qNext * np.cos(gamres) * np.cos(xres)
        deltay = Rq2qNext * np.cos(gamres) * np.sin(xres)
        deltaz = Rq2qNext * np.sin(gamres)

        qNext = q + np.array([deltax, deltay, deltaz])
        return x1, gam1, xres, gamres, qNext

    def line_intersect_circle(self, p, lsp, esp):  # 计算直线和圆的交点
        # p is the circle parameter, lsp and lep is the two end of the line
        x0, y0, r0 = p
        x1, y1 = lsp
        x2, y2 = esp

        x0, y0, r0 = p
        x1, y1 = lsp
        x2, y2 = esp
        x0 = round(x0, 2)
        y0 = round(y0, 2)
        r0 = round(r0, 2)
        x1 = round(x1, 2)
        y1 = round(y1, 2)
        x2 = round(x2, 2)
        y2 = round(y2, 2)

        if r0 == 0:
            return [[x1, y1]]
        if x1 == x2:
            if abs(r0) >= abs(x1 - x0):
                p1 = x1, round(y0 - math.sqrt(r0 ** 2 - (x1 - x0) ** 2), 5)
                p2 = x1, round(y0 + math.sqrt(r0 ** 2 - (x1 - x0) ** 2), 5)
                inp = [p1, p2]
                # select the points lie on the line segment
                inp = [p for p in inp if p[0] >= min(x1, x2) and p[0] <= max(x1, x2)]
            else:
                inp = []
        else:
            k = (y1 - y2) / (x1 - x2)
            b0 = y1 - k * x1
            a = k ** 2 + 1
            b = 2 * k * (b0 - y0) - 2 * x0
            c = (b0 - y0) ** 2 + x0 ** 2 - r0 ** 2
            delta = b ** 2 - 4 * a * c
            if delta >= 0:
                p1x = round((-b - math.sqrt(delta)) / (2 * a), 5)
                p2x = round((-b + math.sqrt(delta)) / (2 * a), 5)
                p1y = round(k * p1x + b0, 5)
                p2y = round(k * p2x + b0, 5)
                inp = [[p1x, p1y], [p2x, p2y]]
                # select the points lie on the line segment
                inp = [p for p in inp if p[0] >= min(x1, x2) and p[0] <= max(x1, x2)]
            else:
                inp = []
        return inp if inp != [] else [[x1, y1]]

    @staticmethod
    def distanceCost(point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def get_vertical_vector(self, vec):
        """ 求二维的向量的垂直向量 """
        assert isinstance(vec, list) and len(vec) == 2, r'平面上的向量必须为2'
        return [vec[1], -vec[0]]

    def initField(self, pos, V0, goal):
        """计算初始流场，返回列向量。"""
        temp1 = pos[0] - goal[0]
        temp2 = pos[1] - goal[1]
        temp3 = pos[2] - goal[2]
        temp4 = self.distanceCost(pos, goal)
        return -np.array([temp1, temp2, temp3], dtype=float).reshape(-1, 1) * V0 / temp4

    @staticmethod
    def partialDerivativeSphere(obs, pos, r):
        """计算球障碍物方程偏导数，返回列向量。"""
        temp1 = pos[0] - obs[0]
        temp2 = pos[1] - obs[1]
        temp3 = pos[2] - obs[2]
        return np.array([temp1, temp2, temp3], dtype=float).reshape(-1, 1) * 2 / r ** 2

    @staticmethod
    def calculateT(obs, pos, r):
        """计算T。"""
        temp1 = pos[0] - obs[0]
        temp2 = pos[1] - obs[1]
        temp3 = pos[2] - obs[2]
        return (temp1 ** 2 + temp2 ** 2 + temp3 ** 2) / r ** 2

    @staticmethod
    def partialDerivativeSphere2(obs, pos, r):
        """计算球障碍物方程偏导数，返回列向量。"""
        temp1 = pos[0] - obs[0]
        temp2 = pos[1] - obs[1]
        temp3 = 0
        return np.array([temp1, temp2, temp3], dtype=float).reshape(-1, 1) * 2 / r ** 2

    @staticmethod
    def calculateT2(obs, pos, r):
        """计算T。"""
        temp1 = pos[0] - obs[0]
        temp2 = pos[1] - obs[1]
        temp3 = 0
        return (temp1 ** 2 + temp2 ** 2 + temp3 ** 3) / r ** 2

    def calPathLen(self, path):
        """计算一个轨迹的长度。"""
        num = path.shape[0]
        len = 0
        for i in range(num - 1):
            len += self.distanceCost(path[i, :], path[i + 1, :])
        return len

    def trans(self, originalPoint, xNew, yNew, zNew):
        """
        坐标变换后地球坐标下坐标
        newX, newY, newZ是新坐标下三个轴上的方向向量
        返回列向量
        """
        lenx = self.calVecLen(xNew)
        cosa1 = xNew[0] / lenx
        cosb1 = xNew[1] / lenx
        cosc1 = xNew[2] / lenx

        leny = self.calVecLen(yNew)
        cosa2 = yNew[0] / leny
        cosb2 = yNew[1] / leny
        cosc2 = yNew[2] / leny

        lenz = self.calVecLen(zNew)
        cosa3 = zNew[0] / lenz
        cosb3 = zNew[1] / lenz
        cosc3 = zNew[2] / lenz

        B = np.array([[cosa1, cosb1, cosc1],
                      [cosa2, cosb2, cosc2],
                      [cosa3, cosb3, cosc3]], dtype=float)

        invB = np.linalg.inv(B)
        return np.dot(invB, originalPoint.T)

    @staticmethod
    def calVecLen(vec):
        """计算向量模长。"""
        return np.sqrt(np.sum(vec ** 2))

    @staticmethod
    def angleVec(vec1, vec2):  # 计算两个向量之间的夹角
        temp = np.dot(vec1, vec2) / np.sqrt(np.sum(vec1 ** 2)) / np.sqrt(np.sum(vec2 ** 2))
        temp = np.clip(temp, -1, 1)  # 可能存在精度误差导致上一步的temp略大于1，因此clip
        theta = np.arccos(temp)
        return theta

    def is_within_perception_range(self, x_range, y_range, pos, target):
        """
        判断目标是否在矩形感知范围内

        :param target_x: 目标的x坐标
        :param target_y: 目标的y坐标
        :param x_min: 矩形感知范围的左边界x坐标
        :param y_min: 矩形感知范围的下边界y坐标
        :param x_max: 矩形感知范围的右边界x坐标
        :param y_max: 矩形感知范围的上边界y坐标

        :return: 如果目标在范围内返回True，否则返回False
        """
        x_min, y_min = pos[0] - x_range, pos[1] - y_range
        x_max, y_max = pos[0] + x_range, pos[1] + y_range
        target_x, target_y = target[0], target[1]
        if x_min <= target_x <= x_max and y_min <= target_y <= y_max:
            return True
        else:
            return False

    def detect_enemy(self, image_path):
        # 读取图片
        image = cv2.imread(image_path)
    
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
        # 设置红色的HSV范围
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
    
        # 使用范围值生成掩码
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2
    
        # 对原图像和掩码进行位运算
        result = cv2.bitwise_and(image, image, mask=mask)
    
        # 检测掩码中是否有非零像素点，非零即代表存在红色
        if cv2.countNonZero(mask) > 0:
            return True
        else:
            return False

    def find_and_label_regions(self, image_path, task, all_opp, all_nei, assign, pos, flag, episode):
        if len(all_opp) != 0 or len(all_nei) != 0:
            # 1. 读取图片
            img = cv2.imread(image_path)
            # 获取图片的高度和宽度
            height, width = img.shape[:2]
            green_x = width // 2
            green_y = height // 2
            # 2. 将图片从 BGR 转换为 HSV
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            if len(all_opp)!= 0:
                # 3. 定义HSV范围
                lower_red = np.array([0, 255, 0])  # 红色低值范围
                upper_red = np.array([10, 255, 255])  # 红色低值范围
                # 4. 使用 cv2.inRange() 创建掩码
                red_mask = cv2.inRange(hsv_img, lower_red, upper_red)
                # 6. 使用 findContours 查找红色区域的轮廓
                contours1, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # 7. 在每个轮廓的中心添加序号
                distance = np.ones([1, len(all_opp)]) * np.inf
                for idx, value in enumerate(all_opp):
                    distance[0][idx] = self.distanceCost(pos[flag - 1], pos[value])
                # 获取排序后的数组及原始索引
                sorted_indices1 = np.argsort(distance[0])  # 获取排序后的索引
                opp_distance = []
                for j in sorted_indices1:
                    opp_distance.append(all_opp[j])  # (距离从小到大的友军序号）
                i = 0
                all_red_x = []
                all_red_y = []
                # print(all_opp)
                # print(opp_distance)
                for idx, contour in enumerate(contours1):
                    if cv2.contourArea(contour) > 0:  # 可以设置一个阈值，忽略小面积的红色区域
                        # 计算轮廓的边界框
                        x, y, w, h = cv2.boundingRect(contour)
                        # 计算文本的中心位置
                        center_x, center_y = x + w // 2, y + h // 2
                        all_red_x.append(center_x)
                        all_red_y.append(center_y)
                        distance[0][i] = self.distanceCost(np.array([center_x, center_y]), np.array([green_x, green_y]))
                        i += 1
                # 获取排序后的数组及原始索引
                sorted_indices2 = np.argsort(distance[0])  # 获取排序后的索引(距离从小到大的红色无人车序号）
                # print(sorted_indices2)
                # print(all_red_x)
                for idx, j in enumerate(sorted_indices2):  # 由近到远画敌军
                    cv2.putText(img, str(idx + 1), (all_red_x[j], all_red_y[j]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if len(all_nei) != 0:
                lower_blue = np.array([100, 255, 255])  # 蓝色低值范围
                upper_blue = np.array([140, 255, 255])  # 蓝色高值范围

                blue_mask = cv2.inRange(hsv_img, lower_blue, upper_blue)

                contours2, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                distance = np.ones([1, len(all_nei)]) * np.inf
                for idx, value in enumerate(all_nei):
                    distance[0][idx] = self.distanceCost(pos[flag-1],pos[value])
                # 获取排序后的数组及原始索引
                sorted_indices1 = np.argsort(distance[0])  # 获取排序后的索引
                teammate_task= []
                nei_distance=[]
                for j in sorted_indices1:
                    nei_distance.append(all_nei[j])  # (距离从小到大的友军序号）
                    teammate_task.append(task[all_nei[j]])  # (距离从小到大的友军任务）
                i = 0
                all_blue_x = []
                all_blue_y = []
                for idx, contour in enumerate(contours2):
                    if cv2.contourArea(contour) > 0:  # 可以设置一个阈值，忽略小面积的蓝色区域
                        # 计算轮廓的边界框
                        x, y, w, h = cv2.boundingRect(contour)
                        # 计算文本的中心位置
                        center_x, center_y = x + w // 2, y + h // 2
                        all_blue_x.append(center_x)
                        all_blue_y.append(center_y)
                        distance[0][i] = self.distanceCost(np.array([center_x, center_y]), np.array([green_x, green_y]))
                        i += 1
                # 获取排序后的数组及原始索引
                sorted_indices2 = np.argsort(distance[0])  # 获取排序后的索引(距离从小到大的蓝色无人车序号）
                for idx, j in enumerate(sorted_indices2): # 由近到远画友军
                    if teammate_task[idx] == -3:
                        cv2.putText(img, str(idx+1)+"search", (all_blue_x[j], all_blue_y[j]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    elif teammate_task[idx] == -2:
                        cv2.putText(img, str(idx + 1) + "escape", (all_blue_x[j], all_blue_y[j]), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (255, 0, 0), 2)
                    elif teammate_task[idx] == -1:
                        cv2.putText(img, str(idx + 1) + "support", (all_blue_x[j], all_blue_y[j]), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (255, 0, 0), 2)
                    elif teammate_task[idx] == 0:
                        cv2.putText(img, str(idx + 1) + "chase", (all_blue_x[j], all_blue_y[j]), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (255, 0, 0), 2)
            cv2.imwrite(image_path, img)

        if 0 <= assign < self.numberofuav/2: # 支援
            output_str = "Supporting. The green UAV is surrounded by %i red UAV (enemy) " % len(all_opp)
            if len(all_nei) == 1:
                output_str += "and %i blue UAV (teammate). " % len(all_nei)
                if teammate_task.count(-3) == 1:
                    output_str += "%i teammate is engaged in searching. " % teammate_task.count(-3)
                if teammate_task.count(-1) == 1:
                    output_str += "%i teammate is providing support. " % teammate_task.count(-1)
                if teammate_task.count(-2) == 1:
                    output_str += "%i teammate is escaping. " % teammate_task.count(-2)
                if teammate_task.count(0) == 1:
                    output_str += "%i teammate is chasing. " % teammate_task.count(0)
            else:
                output_str += "and %i blue UAVs (teammate). " % len(all_nei)
                if teammate_task.count(-3) > 1:
                    output_str += "%i teammates are engaged in searching. " % teammate_task.count(-3)
                elif teammate_task.count(-3) == 1:
                    output_str += "%i teammate is engaged in searching. " % teammate_task.count(-3)
                if teammate_task.count(-1) > 1:
                    output_str += "%i teammates are providing support. " % teammate_task.count(-1)
                elif teammate_task.count(-1) == 1:
                    output_str += "%i teammate is providing support. " % teammate_task.count(-1)
                if teammate_task.count(-2) > 1:
                    output_str += "%i teammates are escaping. " % teammate_task.count(-2)
                elif teammate_task.count(-2) == 1:
                    output_str += "%i teammate is escaping. " % teammate_task.count(-2)
                if teammate_task.count(0) > 1:
                    output_str += "%i teammates are chasing. " % teammate_task.count(0)
                elif teammate_task.count(0) == 1:
                    output_str += "%i teammate is chasing. " % teammate_task.count(0)
            output_str += "Since no enemy forces have been detected, " \
                          "the green UAV should perform a supporting mission. "
            if task[assign] == -2:
                output_str += "Based on the analysis, blue UAV %i is closer and in a escaping state, " \
                          "so it should be prioritized for support." % (nei_distance.index(assign) + 1)
            elif task[assign] == 0:
                output_str += "Based on the analysis, blue UAV %i is closer and in a chasing state, " \
                          "so it should be prioritized for support." % (nei_distance.index(assign) + 1)
            # print("支援 %i 号蓝色无人车" % (nei_distance.index(assign) + 1))
        elif assign >= self.numberofuav/2: # 追击
            if len(all_opp) <= 1:
                output_str = "Chasing. The green UAV is surrounded by %i red UAV (enemy) " % len(all_opp)
            else:
                output_str = "Chasing. The green UAV is surrounded by %i red UAVs (enemy) " % len(all_opp)
            if len(all_nei) <= 1:
                output_str += "and %i blue UAV (teammate). " % len(all_nei)
            else:
                output_str += "and %i blue UAVs (teammate). " % len(all_nei)
            output_str += "Since enemy forces are present and " \
                          "the average velocity direction of our side is superior to that of the enemy, " \
                          "the green UAV should carry out a chasing mission. " \
                          "Based on the analysis, red UAV %i is closer and should be prioritized for pursuit." % (opp_distance.index(assign) + 1)
            # print("追击 %i 号红色无人车" % (opp_distance.index(assign) + 1))
        elif assign == -1: # 逃跑
            if len(all_opp) <= 1:
                output_str = "Escaping. The green UAV is surrounded by %i red UAV (enemy) " % len(all_opp)
            else:
                output_str = "Escaping. The green UAV is surrounded by %i red UAVs (enemy) " % len(all_opp)
            if len(all_nei) <= 1:
                output_str += "and %i blue UAV (teammate). " % len(all_nei)
            else:
                output_str += "and %i blue UAVs (teammate). " % len(all_nei)
            output_str += "Since enemy forces are present and " \
                          "the average velocity direction of our side is inferior to that of the enemy, " \
                          "the green UAV should perform an escaping mission and " \
                          "use a trajectory prediction algorithm to " \
                          "generate the least likely target points where enemies might appear."
            # print("逃跑")
        elif assign == -2: # 搜索
            output_str = "Searching. The green UAV is surrounded by %i red UAV (enemy) " % len(all_opp)
            if len(all_nei) <= 1:
                output_str += "and %i blue UAV (teammate). " % len(all_nei)
                if len(all_nei) == 1:
                    if teammate_task.count(-3) == 1:
                        output_str += "%i teammate is engaged in searching. " % teammate_task.count(-3)
                    if teammate_task.count(-1) == 1:
                        output_str += "%i teammate is providing support. " % teammate_task.count(-1)
            else:
                output_str += "and %i blue UAVs (teammate). " % len(all_nei)
                if teammate_task.count(-3) > 1:
                    output_str += "%i teammates are engaged in searching. " % teammate_task.count(-3)
                elif teammate_task.count(-3) == 1:
                    output_str += "%i teammate is engaged in searching. " % teammate_task.count(-3)
                if teammate_task.count(-1) > 1:
                    output_str += "%i teammates are providing support. " % teammate_task.count(-1)
                elif teammate_task.count(-1) == 1:
                    output_str += "%i teammate is providing support. " % teammate_task.count(-1)

            output_str += "Since no enemy forces have been detected and no teammates are escaping or chasing, " \
                          "the green UAV should perform a searching mission " \
                          "and use a trajectory prediction algorithm to " \
                          "generate the most likely target points where the enemy might appear."
            # print("搜索")
        filename = f"./fig_text/frame-{episode}-@sec.txt"
        with open(filename, "w", encoding="utf-8") as file:
            # 将字符串写入文件
            file.write(output_str)
        print(episode,output_str)

if __name__ == "__main__":
    iifds = IIFDS()
    iifds.loop()
