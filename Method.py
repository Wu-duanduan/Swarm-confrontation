#!/usr/bin/python


import matplotlib.pyplot as plt
import numpy as np
import torch
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class TacticalRewardSystem:
#     def __init__(self):

#         # 权重配置
#         self.weights = {
#             'base_chase': 0.2,
#             'safe_attack_bonus': 2.0,
#             'steer_penalty': -0.015,
#             'predict_penalty': -0.5,
#             'explore_per_grid': 0.1,
#             'escape_penalty_if_safe': -1.5
#         }
        
#         # 状态跟踪
#         self.prev_pose = None
#         self.explored_grids = set()
#         self.enemy_history = {}  # {enemy_id: deque(maxlen=5)}

#     def _relative_angle(self, src_pose, tgt_pos):
#         """计算目标位置相对于源朝向的角度(-180~180)"""
#         dx = tgt_pos[0] - src_pose[0]
#         dy = tgt_pos[1] - src_pose[1]
#         abs_angle = math.degrees(math.atan2(dy, dx))
#         rel_angle = (abs_angle - src_pose[2] + 360) % 360
#         return rel_angle - 360 if rel_angle > 180 else rel_angle

#     def _is_in_sector(self, dist, angle, R, theta):
#         """判断是否在指定扇形区域内"""
#         return dist <= R and abs(angle) <= theta/2

#     def _enemy_threat_level(self, self_pose, enemy):
#         """评估单个敌军的综合威胁"""
#         # 计算相对参数
#         ex, ey, eheading = enemy['pos'][0], enemy['pos'][1], enemy['heading']
#         dist = math.hypot(ex - self_pose[0], ey - self_pose[1])
#         rel_angle = self._relative_angle(self_pose, (ex, ey))
        
#         # 敌方对我方的探测能力
#         enemy_detect_angle = self._relative_angle(
#             (ex, ey, eheading), self_pose[:2]
#         )
#         in_enemy_detect = self._is_in_sector(
#             dist, enemy_detect_angle, self.enemy_R1, self.enemy_theta1
#         )
        
#         # 威胁基值（距离反比 × 角度系数）
#         threat = (1 / (dist + 1e-5)) * (1 - abs(rel_angle)/180)
        
#         # 敌方能力调整
#         if in_enemy_detect:
#             threat *= 2.0 if dist <= self.enemy_R1/2 else 1.5
#             if 'attack' in enemy['status']:  # 假设能获取敌方状态
#                 threat *= 3.0
#         return threat

#     def _predict_enemy_turn(self, enemy_id):
#         """基于历史数据预测敌军转向"""
#         history = self.enemy_history.get(enemy_id, deque(maxlen=5))
#         if len(history) < 2:
#             return False
#         # 计算平均角速度
#         delta = sum(h[1] for h in history)/len(history)
#         return abs(delta) > 15  # 若平均转向速度>15度/步，视为主动搜索

#     def calculate(self, self_pose, enemies, action):
#         """主奖励计算"""
#         reward = 0.0
#         safe_to_attack = True
#         max_threat = 0.0
#         grid = (int(self_pose[0]), int(self_pose[1]))
        
#         #=== 多敌军威胁分析 ===#
#         for enemy in enemies:
#             # 威胁评估
#             threat = self._enemy_threat_level(self_pose, enemy)
#             max_threat = max(max_threat, threat)
            
#             # 更新敌方历史数据
#             enemy_id = enemy['id']
#             rel_angle = self._relative_angle(self_pose, enemy['pos'][:2])
#             self.enemy_history.setdefault(enemy_id, deque(maxlen=5)).append(
#                 (enemy['pos'][2], rel_angle)
#             )
            
#             # 判断是否所有敌军均无威胁
#             if threat > 0.5:  # 经验阈值
#                 safe_to_attack = False
                
#             # 预测转向惩罚
#             if self._predict_enemy_turn(enemy_id):
#                 reward += self.weights['predict_penalty']
        
#         #=== 动作专项逻辑 ===#
#         if action == "chase":
#             # 安全攻击奖励倍增
#             if safe_to_attack and len(enemies) > 0:
#                 reward += self.weights['safe_attack_bonus'] * len(enemies)
#             # 距离奖励
#             if self.prev_pose and len(enemies) > 0:
#                 closest = min(
#                     math.hypot(e['pos'][0]-self_pose[0], e['pos'][1]-self_pose[1])
#                     for e in enemies
#                 )
#                 prev_closest = min(
#                     math.hypot(e['pos'][0]-self.prev_pose[0], e['pos'][1]-self.prev_pose[1])
#                     for e in enemies
#                 )
#                 reward += self.weights['base_chase'] * (prev_closest - closest)
                
#         elif action == "escape":
#             # 安全时逃跑惩罚
#             if safe_to_attack:
#                 reward += self.weights['escape_penalty_if_safe']
#             else:
#                 # 根据最高威胁动态调整
#                 reward += 0.1 * max_threat
        
#         elif action == "search":
#             # 探索奖励
#             if grid not in self.explored_grids:
#                 self.explored_grids.add(grid)
#                 reward += self.weights['explore_per_grid']
        
#         #=== 姿态变化惩罚 ===#
#         if self.prev_pose:
#             delta_steer = abs(self_pose[2] - self.prev_pose[2])
#             reward += self.weights['steer_penalty'] * delta_steer
        
#         self.prev_pose = self_pose
#         return reward

def getReward3(iifds, uavPos, uavVel, task_index, missle_index, all_opp, all_nei, all_nei_c2e):
    """
    获取任务分配奖励值函数
    """
    rewardsum = 0
    base_reward = 1
    base_penalty = -1

    if missle_index[0] == 0: # 弹药为空
        if task_index[0] == -2: # 选择逃逸
            rewardsum = base_reward
        else:
            rewardsum = base_penalty
    else:
        if len(all_opp[0]) != 0:  # 发现敌方
            ave_opp_pos = sum(uavPos[index] for index in all_opp[0]) / len(all_opp[0])
            ave_opp_vel = sum(uavVel[index] for index in all_opp[0]) / len(all_opp[0])
            ave_nei_pos = (sum(uavPos[index] for index in all_nei[0]) + uavPos[0]) / (len(all_nei[0]) + 1)
            ave_nei_vel = (sum(uavVel[index] for index in all_nei[0]) + uavVel[0]) / (len(all_nei[0]) + 1)
            if iifds.cos_cal(ave_nei_vel, ave_opp_pos - ave_nei_pos) >= iifds.cos_cal(ave_opp_vel, -ave_opp_pos + ave_nei_pos):
                if task_index[0] == 0:  # 选择追击
                    rewardsum = 2*base_reward
                else:
                    rewardsum = base_penalty
            else:
                if task_index[0] == -2:  # 选择逃逸
                    rewardsum = base_reward
                else:
                    rewardsum = base_penalty
        else:
            if len(all_nei_c2e[0]) != 0:  # 存在逃跑或追击的友军
                if task_index[0] == -1:  # 选择支援
                    rewardsum = 1.5*base_reward
                else:
                    rewardsum = base_penalty
            else:
                if task_index[0] == -3: # 选择搜索
                    rewardsum = base_reward
                else:
                    rewardsum = base_penalty

    return rewardsum

def getReward2(qNext, obsCenterNext, obs_num, goal, iifds, start):
    """
    获取强化学习奖励值函数
    """
    distance = []
    distances_g = []
    flag_ = 0

    for i in range(int(iifds.numberofuav / 2)):
        i = i + int(iifds.numberofuav / 2)
        distances_g.append(iifds.distanceCost(start[i], goal[i]))
        for j in range(obs_num):
            distance.append(iifds.distanceCost(qNext[i], obsCenterNext[j]))
            if iifds.distanceCost(qNext[i], obsCenterNext[j]) <= iifds.uavR + iifds.obsR:
                flag_ = 1
        k = i + 1
        while k < iifds.numberofuav:
            distance.append(iifds.distanceCost(qNext[i], qNext[k]))
            if iifds.distanceCost(qNext[i], qNext[k]) <= 2 * iifds.uavR:
                flag_ = 1
            k += 1

    rewardsum = 0
    rewarduav = []
    dis_len = len(distance)

    if flag_ == 1:  # 彼此碰撞
        for i in range(dis_len):
            if i < obs_num * (int(iifds.numberofuav / 2)) and distance[i] <= iifds.uavR + iifds.obsR:
                rewardsum += (distance[i] - (iifds.uavR + iifds.obsR)) / (iifds.uavR + iifds.obsR) - 1
            elif i >= obs_num * (int(iifds.numberofuav / 2)) and distance[i] <= 2 * iifds.uavR:
                rewardsum += (distance[i] - (2 * iifds.uavR)) / (2 * iifds.uavR) - 1

    distancegoal = []
    for i in range(int(iifds.numberofuav / 2)):

        distancegoal.append(iifds.distanceCost(qNext[i + int(iifds.numberofuav / 2)], goal[i + int(iifds.numberofuav / 2)]))
    # if flag_b != int(iifds.numberofuav / 2):
    # for i in range(int(iifds.numberofuav / 2)):
        rewarduav.append(-distancegoal[i] / distances_g[i])
    # else:
    #     rewarduav.append(30)

    rewardsum += sum(rewarduav)
    return rewardsum


def getReward1(qNext, obsCenterNext, obs_num, goal, iifds, start):
    """
    获取强化学习奖励值函数
    """
    distance = []
    distances_g = []
    flag_ = 0
    for i in range(int(iifds.numberofuav / 2)):
        distances_g.append(iifds.distanceCost(start[i], goal[i]))
        for j in range(obs_num):
            distance.append(iifds.distanceCost(qNext[i], obsCenterNext[j]))
            if iifds.distanceCost(qNext[i], obsCenterNext[j]) <= iifds.uavR + iifds.obsR:
                flag_ = 1
        k = i + 1
        while k < int(iifds.numberofuav / 2):
            distance.append(iifds.distanceCost(qNext[i], qNext[k]))
            if iifds.distanceCost(qNext[i], qNext[k]) <= 2 * iifds.uavR:
                flag_ = 1
            k += 1

    rewardsum = 0
    rewarduav = []
    dis_len = len(distance)

    if flag_ == 1:  # 彼此碰撞
        for i in range(dis_len):
            if i < obs_num * (int(iifds.numberofuav / 2)) and distance[i] <= iifds.uavR + iifds.obsR:
                rewardsum += (distance[i] - (iifds.uavR + iifds.obsR)) / (iifds.uavR + iifds.obsR) - 1
            elif i >= obs_num * (int(iifds.numberofuav / 2)) and distance[i] <= 2 * iifds.uavR:
                rewardsum += (distance[i] - (2 * iifds.uavR)) / (2 * iifds.uavR) - 1

    distancegoal = []
    for i in range(int(iifds.numberofuav / 2)):
        distancegoal.append(iifds.distanceCost(qNext[i], goal[i]))
    # if (distancegoal[0] > iifds.threshold) or (distancegoal[1] > iifds.threshold):
    for i in range(int(iifds.numberofuav / 2)):
        rewarduav.append(-distancegoal[i] / distances_g[i])
    # else:
    #     rewarduav.append(30)

    rewardsum += sum(rewarduav)
    return rewardsum


def get_reward_multiple(env, qNext, dic):
    """多动态障碍环境获取reward函数"""
    reward = 0
    distance = env.distanceCost(qNext, dic['obsCenter'])
    if distance <= dic['obs_r']:
        reward += (distance - dic['obs_r']) / dic['obs_r'] - 1
    else:
        if distance < dic['obs_r'] + 0.4:
            tempR = dic['obs_r'] + 0.4
            reward += (distance - tempR) / tempR - 0.3
        distance1 = env.distanceCost(qNext, env.goal)
        distance2 = env.distanceCost(env.start, env.goal)
        if distance1 > env.threshold:
            reward += -distance1 / distance2
        else:
            reward += -distance1 / distance2 + 3
    return reward


def drawActionCurve(actionCurveList):
    """
    :param actionCurveList: 动作值列表
    :return: None 绘制图像
    """
    plt.figure()
    for i in range(actionCurveList.shape[1]):
        array = actionCurveList[:, i]
        if i == 0: label = 'row01'
        if i == 1: label = 'sigma01'
        if i == 2: label = 'theta1'
        if i == 3: label = 'row02'
        if i == 4: label = 'sigma02'
        if i == 5: label = 'theta2'
        plt.plot(np.arange(array.shape[0]), array, linewidth=2, label=label)
    plt.title('Variation diagram')
    plt.grid()
    plt.xlabel('time')
    plt.ylabel('value')
    plt.legend(loc='best')


def checkPath(apf):
    sum = 0
    for i in range(apf.path.shape[0] - 1):
        sum += apf.distanceCost(apf.path[i, :], apf.path[i + 1, :])
    for i, j in zip(apf.path, apf.dynamicSphere_Path):
        if apf.distanceCost(i, j) <= apf.dynamicSphere_R:
            print('与障碍物有交点，轨迹距离为：', sum)
            return
    print('与障碍物无交点，轨迹距离为：', sum)


def transformAction(actionBefore, actionBound, actionDim):
    """将强化学习输出的动作映射到指定的动作范围"""
    actionAfter = []
    for i in range(actionDim):
        action_i = actionBefore[i]
        action_bound_i = actionBound[i]
        actionAfter.append((action_i + 1) / 2 * (action_bound_i[1] - action_bound_i[0]) + action_bound_i[0])
    return actionAfter


def test(iifds, actors_cur, dynamicController, arglist, conf):
    """动态单障碍环境测试训练效果"""

    q = []
    qBefore = []
    v = []
    vObs =[]
    start = iifds.start
    goal = iifds.goal

    obsCenter = [np.array([-8.5, -1.5, 0.8]), np.array([-9, -1.5, 0.8]),
                 np.array([-9.5, -1.5, 0.8]), np.array([-10, -1.5, 0.8]), np.array([-4, -1.5, 0.8]),

                 np.array([-4.5, -1.5, 0.8]), np.array([-5, -1.5, 0.8]), np.array([-5.5, -1.5, 0.8]),
                 np.array([-6, -1.5, 0.8]), np.array([-6.5, -1.5, 0.8]), np.array([-7, -1.5, 0.8]),
                 np.array([-7.5, -1.5, 0.8]), np.array([-8, -1.5, 0.8]),  # 序号1

                 np.array([-8.5, -4.5, 0.8]), np.array([-9, -4.5, 0.8]),
                 np.array([-9.5, -4.5, 0.8]), np.array([-10, -4.5, 0.8]), np.array([-4, -4.5, 0.8]),

                 np.array([-4.5, -4.5, 0.8]), np.array([-5, -4.5, 0.8]), np.array([-5.5, -4.5, 0.8]),
                 np.array([-6, -4.5, 0.8]), np.array([-6.5, -4.5, 0.8]), np.array([-7, -4.5, 0.8]),
                 np.array([-7.5, -4.5, 0.8]), np.array([-8, -4.5, 0.8]),  # 序号2

                 np.array([-13.5, 4, 0.8]), np.array([-13.5, 3.5, 0.8]), np.array([-13.5, 3, 0.8]),
                 np.array([-13.5, 2.5, 0.8]), np.array([-13.5, 2, 0.8]), np.array([-13.5, 1.5, 0.8]),  # 序号4

                 np.array([-11.5, -6.5, 0.8]), np.array([-12, -6.5, 0.8]), np.array([-12.5, -6.5, 0.8]),
                 np.array([-13, -6.5, 0.8]), np.array([-13.5, -6.5, 0.8]), np.array([-14, -6.5, 0.8]),  # 序号5

                 np.array([-7.5, 6.5, 0.8]), np.array([-7.5, 7, 0.8]), np.array([-7.5, 7.5, 0.8]),
                 np.array([-7.5, 8, 0.8]),  # 序号6

                 # 序号6#

                 # np.array([7.5, 4, 0.8]), np.array([7.5, 4.5, 0.8]), np.array([7.5, 5, 0.8]),
                 # np.array([7.5, 5.5, 0.8]), np.array([7.5, 6, 0.8]), np.array([7.5, 6.5, 0.8]),  #   序号12

                 np.array([9.5, 6, 0.8]), np.array([10, 6, 0.8]), np.array([10.5, 6, 0.8]),
                 np.array([11, 6, 0.8]), np.array([11.5, 6, 0.8]), np.array([12, 6, 0.8]),  # 序号16

                 # np.array([4, 2, 0.8]), np.array([4, 1.5, 0.8]), np.array([4, 1, 0.8]),
                 # np.array([4, 0.5, 0.8]), np.array([4, 0, 0.8]), np.array([4, -0.5, 0.8]),
                 # np.array([4, -1, 0.8]), np.array([4, -1.5, 0.8]), np.array([4, -2, 0.8]),  #序号13

                 np.array([4, -5.5, 0.8]), np.array([4.5, -5.5, 0.8]), np.array([5, -5.5, 0.8]),
                 np.array([5.5, -5.5, 0.8]), np.array([6, -5.5, 0.8]), np.array([6.5, -5.5, 0.8]),
                 np.array([7, -5.5, 0.8]), np.array([7.5, -5.5, 0.8]), np.array([8, -5.5, 0.8]),  # 序号14

                 np.array([10, -2, 0.8]), np.array([10.5, -2, 0.8]), np.array([11, -2, 0.8]),
                 np.array([11.5, -2, 0.8]), np.array([12, -2, 0.8]), np.array([12.5, -2, 0.8]),
                 np.array([13, -2, 0.8]),  # 序号15

                 np.array([-16, 8, 0.8]), np.array([-16, 8.5, 0.8]), np.array([-16, 9, 0.8]),
                 # np.array([-10.5, 8, 0.8]), np.array([-10.5, 7.5, 0.8]), np.array([-10.5, 7, 0.8]), np.array([-10.5, 6.5, 0.8]),  #序号8

                 np.array([6.5, 2, 0.8]), np.array([6.5, 2.5, 0.8]), np.array([6.5, 3, 0.8]),
                 np.array([6.5, 3.5, 0.8]), np.array([6.5, 4, 0.8]), np.array([6.5, 4.5, 0.8]), np.array([6.5, 5, 0.8]),
                 # 序号11

                 np.array([1, 7, 0.8]), np.array([0.5, 7, 0.8]),
                 np.array([0, 7, 0.8]), np.array([-0.5, 7, 0.8]), np.array([-1, 7, 0.8]),

                 np.array([1, 6.5, 0.8]), np.array([1, 6, 0.8]),
                 np.array([-1, 6.5, 0.8]), np.array([-1, 6, 0.8]),  # 序号9

                 np.array([1, -8.5, 0.8]), np.array([0.5, -8.5, 0.8]),
                 np.array([0, -8.5, 0.8]), np.array([-0.5, -8.5, 0.8]), np.array([-1, -8.5, 0.8]),

                 np.array([1, -7.5, 0.8]), np.array([1, -8, 0.8]),
                 np.array([-1, -7.5, 0.8]), np.array([-1, -8, 0.8]),  # 序号10

                 np.array([-5.5, 3.5, 0.8]), np.array([-6, 3.5, 0.8]),
                 np.array([-6.5, 3.5, 0.8]), np.array([-7, 3.5, 0.8]), np.array([-7.5, 3.5, 0.8]),

                 np.array([-5.5, 3, 0.8]), np.array([-5.5, 2.5, 0.8]),
                 np.array([-7.5, 3, 0.8]), np.array([-7.5, 2.5, 0.8]),  # 序号7

                 # np.array([-14, 3, 0.8]), np.array([-14, 2.5, 0.8]), np.array([-14, 2, 0.8]),
                 # np.array([-14, 1.5, 0.8]), np.array([-14, 1, 0.8]), np.array([-14, 0.5, 0.8]),
                 # np.array([-14, 0, 0.8]),
                 # np.array([-13.5, 2.5, 0.8]), np.array([-13, 2.5, 0.8]), np.array([-12.5, 2.5, 0.8]),
                 # np.array([-13.5, 0.5, 0.8]), np.array([-13, 0.5, 0.8]), np.array([-12.5, 0.5, 0.8]),  #序号3

                 np.array([13.5, 3.5, 0.8]), np.array([13.5, 3, 0.8]),
                 np.array([13.5, 2.5, 0.8]), np.array([13.5, 2, 0.8]), np.array([13.5, 1.5, 0.8]),

                 np.array([13, 3.5, 0.8]), np.array([12.5, 3.5, 0.8]),
                 np.array([13, 1.5, 0.8]), np.array([12.5, 1.5, 0.8]),  # 序号17

                 np.array([14, -5.5, 0.8]), np.array([14, -6, 0.8]),
                 np.array([14, -6.5, 0.8]), np.array([14, -7, 0.8]), np.array([14, -7.5, 0.8]),

                 # np.array([13.5, -5.5, 0.8]), np.array([13, -5.5, 0.8]), np.array([12.5, -5.5, 0.8]),
                 # np.array([13.5, -7.5, 0.8]), np.array([13, -7.5, 0.8]), np.array([12.5, -7.5, 0.8]), # 序号18

                 np.array([0, 0, 0.8])

                 ]
    obs_num = len(obsCenter)
    for i in range(obs_num):
        # obsCenter.append(np.array([obs_init[0][i], obs_init[1][i], obs_init[2][i]]))
        vObs.append(np.array([0, 0, 0], dtype=float))
    iifds.reset(obsCenter)

    for i in range(iifds.numberofuav):
        q.append(start[i])
        qBefore.append([None, None, None])
        v.append((q[i] - q[i]) / iifds.timeStep)
    rewardSum1 = 0
    rewardSum2 = 0
    ta_index = np.ones(iifds.numberofuav) * -3
    dead_index = np.zeros(iifds.numberofuav)
    # assign_index = assign_index.reshape(1, -1)
    ta_index = ta_index.reshape(1, -1)
    dead_index = dead_index.reshape(1, -1)
    flag_uav = np.zeros(iifds.numberofuav)
    for i in range(500):
        goal, ass_index, task_index = iifds.assign(q, v, goal, flag_uav, -1)

        obsCenterNext = obsCenter
        vObsNext = vObs

        obsDicq = iifds.calDynamicState(q, v, obsCenter, vObs, obs_num, goal, flag_uav)  # 相对位置字典
        (obs_n_uav1, obs_n_uav2, obs_n_uav3, obs_n_uav4, obs_n_uav5, obs_n_uav6, obs_n_uav7, obs_n_uav8, obs_n_uav9,
         obs_n_uav10, obs_n_uav11, obs_n_uav12, obs_n_uav13, obs_n_uav14, obs_n_uav15, obs_n_uav16, obs_n_uav17,
         obs_n_uav18, obs_n_uav19, obs_n_uav20) = obsDicq['uav1'], obsDicq['uav2'], obsDicq['uav3'], obsDicq[
            'uav4'], obsDicq['uav5'], obsDicq['uav6'], obsDicq['uav7'], obsDicq['uav8'], obsDicq['uav9'], obsDicq[
            'uav10'], obsDicq['uav11'], obsDicq['uav12'], obsDicq['uav13'], obsDicq['uav14'], obsDicq['uav15'], obsDicq[
            'uav16'], obsDicq['uav17'], obsDicq['uav18'], obsDicq['uav19'], obsDicq['uav20']
        obs_n1 = obs_n_uav1 + obs_n_uav2 + obs_n_uav3 + obs_n_uav4 + obs_n_uav5 + obs_n_uav6 + obs_n_uav7 + obs_n_uav8 + obs_n_uav9 + obs_n_uav10
        obs_n2 = obs_n_uav11 + obs_n_uav12 + obs_n_uav13 + obs_n_uav14 + obs_n_uav15 + obs_n_uav16 + obs_n_uav17 + obs_n_uav18 + obs_n_uav19 + obs_n_uav20

        action_n2 = []
        qNext = []
        vNext = []

        action_n1 = [agent(torch.from_numpy(obs).to(arglist.device, torch.float)).detach().cpu().numpy() \
                     for agent, obs in zip(actors_cur, obs_n1)]
        action_n1 = np.clip(action_n1, arglist.action_limit_min, arglist.action_limit_max)
        action_n1 = action_n1.reshape(-1)

        for j in range(int(iifds.numberofuav / 2)):
            obs_n2_ = torch.as_tensor(obs_n2[j], dtype=torch.float, device=device)
            action_n2_ = dynamicController(obs_n2_).cpu().detach().numpy()
            action_n2.append(transformAction(action_n2_, conf.actionBound, conf.act_dim))

            qNext.append(iifds.getqNext(j, q, v, obsCenter, vObs, action_n1[3 * j], action_n1[3 * j + 1],
                                        action_n1[3 * j + 2], qBefore, goal))
            vNext.append((qNext[j] - q[j]) / iifds.timeStep)

        for j in range(int(iifds.numberofuav / 2)):
            qNext.append(iifds.getqNext(j + int(iifds.numberofuav / 2), q, v, obsCenter, vObs, action_n2[j][0],
                                        action_n2[j][1],
                                        action_n2[j][2], qBefore, goal))
            vNext.append(
                (qNext[j + int(iifds.numberofuav / 2)] - q[j + int(iifds.numberofuav / 2)]) / iifds.timeStep)

        for j in range(iifds.numberofuav):

            if flag_uav[j] == 1:
                qNext[j] = q[j]
                vNext[j] = np.array([0, 0, 0])
            else:
                if ass_index[j] != -1:
                    if iifds.distanceCost(goal[j], q[j]) < iifds.threshold:
                        # finish_uav.append(ass_index[j])
                        flag_uav[ass_index[j]] = 1
                        qNext[int(ass_index[j])] = q[int(ass_index[j])]
                        vNext[int(ass_index[j])] = np.array([0, 0, 0])
                        goal, ass_index, task_index = iifds.assign(q, v, goal, flag_uav, -1)
                else:
                    if iifds.distanceCost(goal[j], q[j]) < iifds.threshold2:
                        goal, ass_index, task_index = iifds.assign(q, v, goal, flag_uav, j)

        rew_n1 = getReward1(qNext, obsCenterNext, obs_num, goal, iifds, start)  # 每个agent使用相同的reward
        rew_n2 = getReward2(qNext, obsCenterNext, obs_num, goal, iifds, start)  # 每个agent使用相同的reward
        rewardSum1 += rew_n1
        rewardSum2 += rew_n2

        qBefore = q
        q = qNext
        v = vNext
        obsCenter = obsCenterNext
        vObs = vObsNext
        # for j in range(int(iifds.numberofuav / 2)):
        #     goal[j] = q[iifds.ass[j]]

        # if flag_c == int(iifds.numberofuav / 2):
        if sum(flag_uav[0:int(iifds.numberofuav/2)]) == int(iifds.numberofuav/2) or sum(flag_uav[int(iifds.numberofuav/2):iifds.numberofuav]) == int(iifds.numberofuav/2):
            break
    return rewardSum1, rewardSum2


def setup_seed(seed):
    """设置随机数种子函数"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
