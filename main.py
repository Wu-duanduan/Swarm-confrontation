#!/usr/bin/python

# 模拟UAV（无人车）群体对抗的任务，主要涉及多个无人车（包括红方、蓝方、绿色）在模拟环境中执行任务、攻击和逃避，并且有路径规划和奖励机制。
# 在当前细窄长墙体场景下，路径规划算法效果不是特别理想。关键函数为 iifds.detect，iifds.assign，iifds.getvNext。
# assign函数放了其中一种任务分配例子用于理解。

# 引入了许多常用的库，如 torch、numpy、matplotlib 等，用于深度学习、数值计算、图形绘制等。
import torch
import numpy as np
import matplotlib.pyplot as plt
from All_fun import IIFDS
from Method import getReward1, getReward2
from arguments import parse_args
import random
from config import Config
import argparse
from battle import Battle
import pandas as pd
import pyglet
from pyglet import image
from PIL import Image as PILImage, ImageDraw
import time

seed = 10
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# time.sleep(2)
def get_args():
    parser = argparse.ArgumentParser("UAV swarm confrontation")
    iifds = IIFDS()
    # Environment
    parser.add_argument("--num-RCARs", type=int, default=5, help="number of red CARs")
    parser.add_argument("--num-BCARs", type=int, default=5, help="number of blue CARs")
    parser.add_argument("--num-BUAVs", type=int, default=1, help="number of blue UAVs")

    parser.add_argument("--detect-range", type=float, default=iifds.R_1, help="")

    parser.add_argument("--attack-range-B", type=float, default=iifds.threshold, help="")
    parser.add_argument("--attack-range-R", type=float, default=iifds.threshold, help="")
    parser.add_argument("--attack-angle-BR", type=float, default=iifds.hit_angle / 2, help="")

    parser.add_argument("--sensor-range-B-l", type=float, default=16, help="")
    parser.add_argument("--sensor-range-B-w", type=float, default=12, help="")
    parser.add_argument("--sensor-angle-B", type=float, default=np.pi, help="")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()  # 使用 argparse 来解析命令行参数，配置Python画图的参数。
    env = Battle(args)  # Battle 类实例化环境，调用 env.reset() 重置环境状态。
    env.reset()
    iifds = IIFDS()  # 使用 IIFDS 来处理无人机行为的细节（例如任务分配、路径规划等）。
    conf = Config()
    arglist = parse_args()
    # 加载预训练的模型（Actor），用于无人机的路径规划。
    actors_cur1 = [None for _ in range(int(iifds.numberofuav / 2))]
    actors_cur2 = [None for _ in range(int(iifds.numberofuav / 2))]
    for i in range(int(iifds.numberofuav / 2)):
        actors_cur1[i] = torch.load('TrainedModel/Actor.%d.pkl' % i, map_location=device)
        actors_cur2[i] = torch.load('TrainedModel/Actor.%d.pkl' % i, map_location=device)
    # 初始化无人机的当前位置q，上一位置qBefore，当前速度v，初始位置start，目标goal，障碍物位置obsCenter，障碍物速度Vobs。
    q = []
    qBefore = []
    v = []

    start = iifds.start  # 当前为三维坐标，若考虑二维空间，则将第三维坐标固定。
    goal = iifds.goal
    obsCenter = pd.read_csv('./MADDPG_data_csv/obsCenter.csv').to_numpy()
    vObs = []

    obs_num = len(obsCenter)
    for i in range(obs_num):
        # obsCenter[i][0] = obsCenter[i][0] - 10.85
        # obsCenter[i][1] = obsCenter[i][1] - 7.45
        # obsCenter[i][2] = obsCenter[i][2] * 2 / 3
        vObs.append(np.array([0, 0, 0], dtype=float))  # 设置为静态障碍物。
    # np.savetxt('./MADDPG_data_csv/obsCenter.csv', obsCenter, delimiter=',')
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
    dead_index = flag_uav.reshape(1, -1)  # 存放所有时刻各无人机的存活情况
    total_missle_index = missle_index.reshape(1, -1)  # 表示所有时刻各无人机的血量剩余情况
    total_HP_index = HP_index.reshape(1, -1)  # 表示所有时刻各无人机的血量剩余情况

    pos_uav = []
    vel_uav = []
    pos_uav.append(np.array([0, 0, 0.8*2/3], dtype=float))
    vel_uav.append(np.array([0, 0, 0], dtype=float))

    observe_agent = -1  # 设置需要观察的无人机序号，0表示全局模式，1-10分别为每个单独的无人车序号，-1表示无人机视野
    for i in range(300):
        for j in range(len(pos_uav)):
            vel_uav[j][0] = random.random() * 2 - 1
            vel_uav[j][1] = random.random() * 2 - 1
            pos_uav[j] += vel_uav[j] * iifds.timeStep
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
        # 检测在无人机群体中每个无人机感知半径内的敌方、友方、正在追逐或者逃跑的友方，与其最接近的敌方和友方的序号。
        all_opp, all_nei, all_nei_c2e, all_close_opp, all_close_nei = iifds.detect(q, flag_uav, ta_index, HP_index,
                                                                                   obsCenter)

        # 根据感知信息进行任务分配，goal为分配后的各无人机目标位置，ass_index为追击或支援无人机分配的目标序号，task_index为任务信息。
        goal, ass_index, task_index = iifds.assign(q, v, goal, missle_index, i,
                                                   pos_b, pos_r, ta_index, obsCenter, all_opp, all_nei, all_nei_c2e,
                                                   all_close_opp, all_close_nei)

        ta_index = np.vstack((ta_index, task_index))
        dead_index = np.vstack((dead_index, flag_uav))
        total_missle_index = np.vstack((total_missle_index, missle_index))
        total_HP_index = np.vstack((total_HP_index, HP_index))

        # 路径规划
        obsCenterNext = obsCenter
        vObsNext = vObs
        # 根据当前位置、目标位置、障碍物位置，规划避障路径，输出为下一时刻的速度矢量（包括大小和方向）
        vNext = iifds.getvNext(q, v, obsCenter, vObs, qBefore, goal, flag_uav, arglist, actors_cur1, actors_cur2)
        # print(vNext)
        # 根据一阶积分计算下一时刻位置
        qNext = []
        for j in range(iifds.numberofuav):
            qNext.append(q[j] + vNext[j] * iifds.timeStep)

        # 计算伤亡情况
        for j in range(iifds.numberofuav):
            if flag_uav[j] == 1:
                qNext[j] = q[j]
                vNext[j] = np.array([0, 0, 0])
            else:
                if task_index[j] == 0:  # 如果是追击
                    if iifds.distanceCost(goal[j], q[j]) < iifds.threshold and iifds.cos_cal(goal[j] - q[j],
                                                                                             v[j]) < np.cos(
                            iifds.hit_angle / 2):  # 目标小于开火范围
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

        rew_n1 = getReward1(qNext, obsCenterNext, obs_num, goal, iifds, start)  # 每个agent使用相同的路径reward
        rew_n2 = getReward2(qNext, obsCenterNext, obs_num, goal, iifds, start)

        env.render(q, v, iifds.R_1, all_opp[observe_agent - 1], all_nei[observe_agent - 1], total_HP_index[-1],
                   iifds.HP_num, total_missle_index[-1] / 2, iifds.missle_num / 2, observe_agent,
                   task_index, pos_uav, vel_uav)  # 画出上一时刻的位置速度、血量、弹药
        # print(q[1])
        if i % 15 == 0 and i != 0:
            try:
                color_buffer = pyglet.image.get_buffer_manager().get_color_buffer()

                # 获取图像的原始像素数据
                image_data = color_buffer.get_image_data()

                # 将数据转换为 NumPy 数组（需要转换为 RGB 格式）
                img_data = np.frombuffer(image_data.get_data('RGB', image_data.width * 3), dtype=np.uint8)
                img_data = img_data.reshape((image_data.height, image_data.width, 3))

                # 翻转图像的垂直方向
                img_data = np.flipud(img_data)  # 或者 img_data = img_data[::-1]

                # 使用 Pillow 保存图像
                img = PILImage.fromarray(img_data)

                filename = f"./fig_text/frame-{i}-@sec.png"
                img.save(filename)
                if observe_agent > 0:
                    # try:
                    iifds.find_and_label_regions(filename, ta_index[-1], all_opp[observe_agent - 1],
                                                 all_nei[observe_agent - 1], ass_index[observe_agent - 1], q,
                                                 observe_agent, i)  # 存储上一时刻的序号以及友军任务情况照片
            except Exception as e:
                # pass
                print("error!")
        qBefore = q
        q = qNext
        v = vNext
        obsCenter = obsCenterNext
        vObs = vObsNext

        # 信息保存
        for j in range(iifds.numberofuav):
            path_var = globals().get(f'path{j + 1}')
            goal_var = globals().get(f'goal{j + 1}')

            path_var = np.vstack((path_var, q[j][0:2]))
            goal_var = np.vstack((goal_var, goal[j][0:2]))

            # 更新全局变量
            globals()[f'path{j + 1}'] = path_var
            globals()[f'goal{j + 1}'] = goal_var
        ta_index = np.vstack((ta_index, task_index))
        dead_index = np.vstack((dead_index, flag_uav))
        total_missle_index = np.vstack((total_missle_index, missle_index))
        total_HP_index = np.vstack((total_HP_index, HP_index))

        # 对抗结束判断
        if sum(flag_uav[0:int(iifds.numberofuav / 2)]) == int(iifds.numberofuav / 2) or sum(
                flag_uav[int(iifds.numberofuav / 2):iifds.numberofuav]) == int(iifds.numberofuav / 2):
            break
        if observe_agent != 0 and flag_uav[observe_agent - 1] == 1:
            break
    for i in range(1, iifds.numberofuav + 1):
        np.savetxt(f'./MADDPG_data_csv/pathMatrix{i}.csv', globals()[f'path{i}'], delimiter=',')
        np.savetxt(f'./MADDPG_data_csv/goalMatrix{i}.csv', globals()[f'goal{i}'], delimiter=',')
    np.savetxt('./MADDPG_data_csv/ass_index.csv', ta_index, delimiter=',')
    np.savetxt('./MADDPG_data_csv/dead_index.csv', dead_index, delimiter=',')
    np.savetxt('./MADDPG_data_csv/total_missle_index.csv', total_missle_index, delimiter=',')
    np.savetxt('./MADDPG_data_csv/total_HP_index.csv', total_HP_index, delimiter=',')
    # np.savetxt('./MADDPG_data_csv/obsCenter.csv', obsCenter, delimiter=',')
    plt.show()
