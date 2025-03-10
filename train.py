import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from All_fun import IIFDS
from Method import getReward3
from rl_utils import ReplayBuffer
from arguments import parse_args
import random
from config import Config
import argparse
from battle import Battle
from RL_brain import DQN
import pandas as pd
import time
import os
seed = random.randint(1, 1000)
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
time.sleep(2)

def get_args():
    parser = argparse.ArgumentParser("CAR swarm confrontation")
    iifds = IIFDS()
    # Environment
    parser.add_argument("--num_RCARs", type=int, default=5, help="number of red CARs")
    parser.add_argument("--num_BCARs", type=int, default=5, help="number of blue CARs")

    parser.add_argument("--detect-range", type=float, default=iifds.R_1, help="")
    parser.add_argument("--attack-range-B", type=float, default=iifds.threshold, help="")
    parser.add_argument("--attack-range-R", type=float, default=iifds.threshold, help="")

    parser.add_argument("--attack-angle-BR", type=float, default=iifds.hit_angle/2, help="")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()   # 使用 argparse 来解析命令行参数，配置Python画图的参数。
    env = Battle(args)  # Battle 类实例化环境，调用 env.reset() 重置环境状态。
    iifds = IIFDS() 
    conf = Config()     # 强化学习参数配置
    arglist = parse_args()
    maxReward = -np.inf
    state_dim = iifds.obs_dim
    action_dim = len(iifds.action_space)
    hidden_dim = conf.hidden_dim
    lr = conf.lr
    gamma = conf.gamma
    epsilon = conf.epsilon
    target_update = conf.target_update
    memory = ReplayBuffer(conf.buffer_size)

    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)

    # 障碍物设置
    obsCenter = pd.read_csv('./MADDPG_data_csv/obsCenter.csv').to_numpy()
    vObs = []
    obs_num = len(obsCenter)
    for i in range(obs_num):
        vObs.append(np.array([0, 0, 0], dtype=float)) 

    # 路径规划网络
    actors_cur1 = [None for _ in range(int(iifds.numberofuav / 2))]
    actors_cur2 = [None for _ in range(int(iifds.numberofuav / 2))]
    for i in range(int(iifds.numberofuav / 2)):
        actors_cur1[i] = torch.load('TrainedModel/Actor.%d.pkl' % i, map_location=device)
        actors_cur2[i] = torch.load('TrainedModel/Actor.%d.pkl' % i, map_location=device)

    reward_list= []
    observe_agent = 1
    for episode_gone in range(conf.MAX_EPISODE): # 开始训练
        env.reset() # 重置环境
        iifds = IIFDS()      
        start = iifds.start 
        goal = iifds.goal
        episode_reward = 0
        step = 0
        done = False

        q = []
        qBefore = []
        v = []    
        for i in range(iifds.numberofuav):
            q.append(start[i])
            qBefore.append([None, None, None])
            v.append((q[i] - q[i]) / iifds.timeStep)  # 初始速度置为0。
        # 使用 globals() 将每个无人机的路径和目标动态赋值给 pathX 和 goalX。
        path = []
        target = []
        for i in range(iifds.numberofuav):
            path.append(start[i][0:2].reshape(1, -1))
            target.append(start[i][0:2].reshape(1, -1))
        # 将 path 和 goal 转换为单独的变量
        for i in range(iifds.numberofuav):
            globals()[f'path{i + 1}'] = path[i]
            globals()[f'goal{i + 1}'] = target[i]
        ta_index = np.ones(iifds.numberofuav) * -3  # 表示当前时刻各无人机的任务目标，-3表示搜索，-2表示逃逸，-1表示支援，0表示追击
        flag_uav = np.zeros(iifds.numberofuav)  # 表示当前时刻各无人机的存活情况，0表示存活，1表示死亡
        missle_index = np.ones(iifds.numberofuav) * iifds.missle_num  # 存放每一轮各无人机的子弹剩余数量
        fill_index = np.zeros(iifds.numberofuav)  # 存放所有时刻各无人机的子弹填充情况
        flag_fill = np.zeros(iifds.numberofuav)  # 表示当前时刻子弹是否填充完成，填充完毕为1，否则为0
        HP_index = np.ones(iifds.numberofuav) * iifds.HP_num  # 表示当前时刻各无人机的血量剩余情况
        ta_index = ta_index.reshape(1, -1)  # 存放所有时刻各无人机的任务目标

        # 初始感知信息
        FOV = np.arctan(0.5) * 2
        present_ids = env.render_BEV(q, v, iifds.R_1, FOV, observe_agent)     # 视野中车辆的编号
        alive_cars = [car_id for car_id in present_ids if flag_uav[car_id] == 0]  # 视野中的存活车辆

        # 构建固定长度的输入向量（10辆车 × 4个特征 = 40维）
        state = []
        next_state = []
        for car_id in range(10):  # 遍历所有车辆0-9
            if car_id in alive_cars:
            # 提取x和y方向的位置和速度（忽略z轴）
                q_x, q_y, _ = q[car_id] 
                v_x, v_y, _ = v[car_id] 
                state.extend([q_x, q_y, v_x, v_y])
            else:
            # 填充4个零（x位置, y位置, x速度, y速度）
                state.extend([0.0, 0.0, 0.0, 0.0])

        # 转换为NumPy数组（形状为40,）
        state = np.array(state, dtype=np.float32)

        for i in range(conf.MAX_STEP):
            # 路径拼接
            pos_b = []
            pos_r = []
            # 创建 pos_b_all 和 pos_r_all
            pos_b_all = [[j + 1, globals()[f'path{j + 1}']] for j in range(5)]  # 蓝队路径
            pos_r_all = [[j + 1, globals()[f'path{j + 6}']] for j in range(5)]  # 红队路径

            # 保存所有存放无人机的路径点用于轨迹预测
            for j in range(iifds.numberofuav):
                if flag_uav[j] == 0:
                    if j < iifds.numberofuav / 2:
                        pos_b.append(pos_b_all[j])
                    else:
                        pos_r.append(pos_r_all[j - int(iifds.numberofuav / 2)])


            # 其他车辆的感知与任务分配
            all_opp, all_nei, all_nei_c2e, all_close_opp, all_close_nei = iifds.detect(q, flag_uav, ta_index, HP_index, obsCenter)
            ave_opp_pos, ave_opp_vel, ave_nei_pos, ave_nei_vel, num_opp, num_nei_c2e = iifds.Meanfield(q, v, all_opp, all_nei, all_nei_c2e)
            task_index_blue = iifds.stateSelectionBlue(ave_opp_pos, ave_opp_vel, ave_nei_pos, ave_nei_vel, num_opp, num_nei_c2e, missle_index)
            task_index_red = iifds.stateSelectionRed(q, v, missle_index, all_opp, all_nei_c2e, all_close_opp)
            task_index = task_index_blue + task_index_red

            # 根据感知信息进行任务分配(actions)
            action = agent.take_action(state)
            task_index[0] = action - 3   
            goal, ass_index = iifds.allocation(q, goal, pos_b, pos_r, ta_index, obsCenter, all_close_opp, all_close_nei, task_index, i)
            ta_index = np.vstack((ta_index, task_index))

            # reward
            reward = getReward3(iifds, q, v, task_index, missle_index, all_opp, all_nei, all_nei_c2e)   
            episode_reward += reward
            
            # 更新UAV(step)
            vNext = iifds.getvNext(q, v, obsCenter, vObs, qBefore, goal, flag_uav, arglist, actors_cur1, actors_cur2)
            qNext = []
            for j in range(iifds.numberofuav):
                qNext.append(q[j] + vNext[j] * iifds.timeStep)
            
            for j in range(iifds.numberofuav): 
                if flag_uav[j] == 1:
                    qNext[j] = q[j]
                    vNext[j] = np.array([0, 0, 0])
                else:
                    if task_index[j] == 0:  # 如果是追击
                        if iifds.distanceCost(goal[j], q[j]) < iifds.threshold and iifds.cos_cal(goal[j]-q[j], v[j]) >- np.cos(iifds.hit_angle/2):  # 目标小于开火范围
                            if random.random() < iifds.hit_rate:
                                if HP_index[ass_index[j]] > 0:
                                    HP_index[ass_index[j]] -= 1
                                    missle_index[j] -= 1
                                    if HP_index[ass_index[j]] == 0:
                                        flag_uav[ass_index[j]] = 1
                            else:
                                missle_index[j] -= 1
                    if missle_index[j] == 0:  # 若子弹未填充完毕，则继续填充且任务会一直设置为逃逸
                        flag_fill[j] = 1
                    if flag_fill[j] == 1:
                        fill_index[j] += iifds.vel_fill_missle
                    if fill_index[j] == iifds.missle_num:  # 若子弹填充完毕，则可以继续参与打击
                        flag_fill[j] = 0
                        fill_index[j] = 0
                        missle_index[j] = iifds.missle_num

            qBefore = q
            q = qNext
            v = vNext
            step += 1

            present_ids = env.render_BEV(q, v, iifds.R_1, FOV, observe_agent) # 视野中车辆的编号
            alive_cars = [car_id for car_id in present_ids if flag_uav[car_id] == 0]  # 视野中的存活车辆

            # 构建固定长度的输入向量（10辆车 × 4个特征 = 40维）
            next_state = []
            for car_id in range(10):  # 遍历所有车辆0-9
                if car_id in alive_cars:
                # 提取x和y方向的位置和速度（忽略z轴）
                    q_x, q_y, _ = q[car_id] 
                    v_x, v_y, _ = v[car_id]  
                    next_state.extend([q_x, q_y, v_x, v_y])
                else:
                # 填充4个零（x位置, y位置, x速度, y速度）
                    next_state.extend([0.0, 0.0, 0.0, 0.0])

            # 转换为NumPy数组（形状为40,）
            next_state = np.array(next_state, dtype=np.float32)

            # 对抗结束判断(done)
            if flag_uav[0] == 1 :    # 阵亡
                done = True
            elif sum(flag_uav[int(iifds.numberofuav / 2):iifds.numberofuav]) == int(iifds.numberofuav / 2): # 红车全部阵亡
                done = True

            # 存入replay_buffer
            memory.add(state, action, reward, next_state, done)
            state = next_state

            # 当buffer数据的数量超过一定值后,才进行Q网络训练
            if memory.size() > conf.minimal_size:
                b_s, b_a, b_r, b_ns, b_d = memory.sample(conf.batch_size)
                transition_dict = {
                    'states': b_s,
                    'actions': b_a,
                    'next_states': b_ns,
                    'rewards': b_r,
                    'dones': b_d
                    }
                agent.update(transition_dict)

            if done:
                break

        # if episode_gone > conf.MAX_EPISODE * 2 / 3:
        if episode_reward > maxReward:
            maxReward = episode_reward
            print("历史最优reward，已保存模型！")
            torch.save({
                'q_net': agent.q_net.state_dict(),
                'best_reward': maxReward
            }, f"Assign_Model/best_model_{episode_reward:.1f}.pth")

        reward_list.append(episode_reward)  # 后续可以用来作图