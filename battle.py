import numpy as np
from math import cos, sin, tan, pi, atan2, acos
from random import random, uniform, randint
from gym import spaces
import pygame
import rendering


class Model(object):
    def __init__(self, args):
        super(Model, self).__init__()
        # 标识
        self.id = None
        self.enemy = None
        self.size = 0.3
        self.color = None
        # 状态
        self.pos = np.zeros(2)
        self.speed = 0  # 速度标量
        self.yaw = 0  # 偏航角，0为x轴正方向，逆时针为正，(-pi, pi)
        # 约束
        # 对抗相关
        self.death = False
        # 感知范围和攻击范围
        self.attack_range = 0
        self.attack_angle = 0
        self.sensor_range_l = 0
        self.sensor_range_w = 0
        self.sensor_angle = 0


class Battle(object):
    """为避免神经网络输入数值过大，采用等比例缩小模型"""

    def __init__(self, args):
        super(Battle, self).__init__()
        self.args = args
        self.dt = 1  # simulation interval，1 second
        self.t = 0
        self.render_geoms = None
        self.render_geoms_xform = None
        self.num_CARs = args.num_RCARs + args.num_BCARs
        self.num_UAVs = args.num_BUAVs
        self.num_RCARs = args.num_RCARs
        self.num_BCARs = args.num_BCARs
        self.num_BUAVs = args.num_BUAVs
        self.CARs = [Model(args) for _ in range(self.num_CARs)]
        self.UAVs = [Model(args) for _ in range(self.num_UAVs)]
        self.RCARs = []
        self.BCARs = []
        self.BUAVs = []
        for i, CAR in enumerate(self.CARs):
            CAR.id = i
            if i < args.num_BCARs:
                CAR.enemy = False
                CAR.color = np.array([0, 0, 1])
                CAR.attack_range = args.attack_range_B
                CAR.attack_angle = args.attack_angle_BR
                self.BCARs.append(CAR)
            elif i < args.num_RCARs + args.num_BCARs:
                CAR.enemy = True
                CAR.color = np.array([1, 0, 0])
                CAR.attack_range = args.attack_range_R
                CAR.attack_angle = args.attack_angle_BR
                self.RCARs.append(CAR)
        for i, UAV in enumerate(self.UAVs):
            UAV.id = i
            UAV.enemy = False
            UAV.color = np.array([0, 0, 1])
            UAV.sensor_range_l = args.sensor_range_B_l
            UAV.sensor_range_w = args.sensor_range_B_w
            UAV.sensor_angle = args.sensor_angle_B
            self.BUAVs.append(UAV)
        self.sensor_range_l = args.sensor_range_B_l
        self.sensor_range_w = args.sensor_range_B_w
        self.viewer = None
        self.action_space = []
        self.reset()

    def reset(self):
        self.t = 0
        # reset render
        self.render_geoms = None
        self.render_geoms_xform = None
        random_side = randint(0, 1)
        for i, CAR in enumerate(self.CARs):
            CAR.being_attacked = False
            CAR.death = False
        for i, UAV in enumerate(self.UAVs):
            UAV.being_attacked = False
            UAV.death = False
            # if not UAV.enemy:
            #     interval = 2.0 / (self.num_RUAVs + 1)
            #     UAV.pos = np.array([random_side * 1.8 - 0.9, 1 - (i + 1) * interval])
            #     UAV.yaw = pi * random_side
            # else:
            #     interval = 2.0 / (self.num_BUAVs + 1)
            #     UAV.pos = np.array([(1 - random_side) * 1.8 - 0.9, 1 - (i - self.num_RUAVs + 1) * interval])
            #     UAV.yaw = pi * (1 - random_side)

    def render(self, pos, vel, detect_range, all_opp, all_nei, HP_index, HP_num, missle_index, missle_num, flag, task,
               mode='rgb_array'):
        pos_copy = np.copy(pos)
        vel_copy = np.copy(vel)

        if self.viewer is None:
            self.viewer = rendering.Viewer(900, 480)
            pygame.init()
        # 每次渲染时清除旧的几何对象
        self.render_geoms = []
        self.render_geoms_xform = []
        # 初始化pygame用于文本渲染

        for i, CAR in enumerate(self.CARs):  # 添加无人车以及攻击范围
            if i == flag - 1:
                CAR.color = np.array([0, 1, 0])
            xform = rendering.Transform()
            for x in rendering.make_CAR(CAR.size):
                x.set_color(*CAR.color)
                x.add_attr(xform)
                self.render_geoms.append(x)
                self.render_geoms_xform.append(xform)

            # render attack range
            sector = rendering.make_sector(radius=CAR.attack_range, theta=2 * CAR.attack_angle)
            sector.set_color(*CAR.color, 0.2)
            sector.add_attr(xform)
            self.render_geoms.append(sector)
            self.render_geoms_xform.append(xform)

            # if flag == 0:
            #     if task[i] == -3:
            #         color_temp = np.array([0.5, 0.5, 0.5])  # 灰色
            #     elif task[i] == -2:
            #         color_temp = np.array([1.0, 1.0, 0.0])  # 黄色
            #     elif task[i] == -1:
            #         color_temp = np.array([0.5, 0.0, 0.5])  # 紫色
            #     elif task[i] == 0:
            #         color_temp = np.array([0, 1, 0])  # 绿色
            #     sector = rendering.make_circle(radius=0.5)
            #     sector.set_color(*color_temp, 0.4)
            #     sector.add_attr(xform)
            #     self.render_geoms.append(sector)
            #     self.render_geoms_xform.append(xform)
            # elif flag > 0:
            #     if task[i] == -3:
            #         color_temp = np.array([0.5, 0.5, 0.5])  # 灰色
            #     elif task[i] == -2:
            #         color_temp = np.array([1.0, 1.0, 0.0])  # 黄色
            #     elif task[i] == -1:
            #         color_temp = np.array([0.5, 0.0, 0.5])  # 紫色
            #     elif task[i] == 0:
            #         color_temp = np.array([0, 1, 0])  # 绿色
            #     if i in all_nei:
            #         sector = rendering.make_circle(radius=0.5)
            #     else:
            #         sector = rendering.make_circle(radius=0)
            #     sector.set_color(*color_temp, 0.4)
            #     sector.add_attr(xform)
            #     self.render_geoms.append(sector)
            #     self.render_geoms_xform.append(xform)
        self.length_temp1 = len(self.render_geoms)

        # # 动态绘制血条
        # health_bar_width = 0.5  # 每个格子的宽度
        # health_bar_height = 0.1  # 格子的高度
        # max_health = HP_num  # 假设血量最大为 100
        # num_cells = HP_num  # 假设血条由 10 个格子组成
        #
        # # 创建血条的位置变换
        # health_xform = rendering.Transform()
        #
        # # 动态绘制血条格子
        # for j in range(num_cells):
        #     # 每个格子的位置
        #     x_offset = j * health_bar_width - 0.5  # 水平偏移，使得格子居中
        #     health_bar = rendering.FilledPolygon([
        #         (x_offset, 0),  # 左下角
        #         (x_offset + health_bar_width, 0),  # 右下角
        #         (x_offset + health_bar_width, health_bar_height),  # 右上角
        #         (x_offset, health_bar_height)  # 左上角
        #     ])
        #     if flag == 0:
        #         if HP_index[i] == 0:
        #             health_bar.set_color(1, 1, 1, 0)  # 白色表示无血量
        #         elif j < HP_index[i] / (max_health / num_cells):  # 计算应该显示的格子数量
        #             health_bar.set_color(255/255, 165/255, 0)  # 红色表示有血量
        #         else:
        #             health_bar.set_color(0.5, 0.5, 0.5)  # 灰色表示无血量
        #     else:
        #         if HP_index[i] == 0 or ((i != flag - 1) and (i not in all_opp) and (i not in all_nei)):
        #             health_bar.set_color(1, 1, 1, 0)  # 白色表示无血量
        #         elif j < HP_index[i] / (max_health / num_cells):  # 计算应该显示的格子数量
        #             health_bar.set_color(255/255, 165/255, 0)  # 红色表示有血量
        #         else:
        #             health_bar.set_color(0.5, 0.5, 0.5)  # 灰色表示无血量
        #     if HP_index[i] != 0:
        #         # 为血条设置位置和旋转
        #         health_bar.add_attr(health_xform)
        #         # 设置血条的 Y 坐标偏移，使其在 UAV 上方
        #         health_xform.set_translation(pos_copy[i][0] - 0.5, pos_copy[i][1] + UAV.size + 0.5)  # 偏移 0.3 让血条在 UAV 上方
        #     self.render_geoms.append(health_bar)
        #     self.render_geoms_xform.append(xform)
        #
        # # 动态绘制弹药
        # missle_bar_width = 0.5  # 每个格子的宽度
        # missle_bar_height = 0.1  # 格子的高度
        # max_missle = missle_num  # 假设弹药最大为 100
        # num_cells = int(missle_num)  # 假设弹药由 10 个格子组成
        #
        # # 创建弹药的位置变换
        # missle_xform = rendering.Transform()
        #
        # # 动态绘制弹药格子
        # for j in range(num_cells):
        #     # 每个格子的位置
        #     x_offset = j * missle_bar_width - 0.5  # 水平偏移，使得格子居中
        #     missle_bar = rendering.FilledPolygon([
        #         (x_offset, 0),  # 左下角
        #         (x_offset + missle_bar_width, 0),  # 右下角
        #         (x_offset + missle_bar_width, missle_bar_height),  # 右上角
        #         (x_offset, missle_bar_height)  # 左上角
        #     ])
        #     if flag == 0:
        #         if missle_index[i] == 0 or HP_index[i] == 0:
        #             missle_bar.set_color(1, 1, 1, 0)  # 白色表示无弹药
        #         elif j < missle_index[i] / (max_missle / num_cells):  # 计算应该显示的格子数量
        #             missle_bar.set_color(0, 1, 0)  # 绿色表示有弹药
        #         else:
        #             missle_bar.set_color(0.5, 0.5, 0.5)  # 灰色表示无弹药
        #     else:
        #         if missle_index[i] == 0 or HP_index[i] == 0 or (
        #                 (i != flag - 1) and (i not in all_opp) and (i not in all_nei)):
        #             missle_bar.set_color(1, 1, 1, 0)  # 白色表示无弹药
        #         elif j < missle_index[i] / (max_missle / num_cells):  # 计算应该显示的格子数量
        #             missle_bar.set_color(0, 1, 0)  # 绿色表示有弹药
        #         else:
        #             missle_bar.set_color(0.5, 0.5, 0.5)  # 灰色表示无弹药
        #     if missle_index[i] != 0:
        #         # 为弹药设置位置和旋转
        #         missle_bar.add_attr(missle_xform)
        #         # 设置弹药的 Y 坐标偏移，使其在 UAV 上方
        #         missle_xform.set_translation(pos_copy[i][0] - 0.5, pos_copy[i][1] + UAV.size + 0.3)  # 偏移 0.3 让弹药在 UAV 上方
        #     self.render_geoms.append(missle_bar)
        #     self.render_geoms_xform.append(xform)

        # 渲染静态障碍物
        self.render_static_obstacles()
        self.viewer.geoms = []
        for geom in self.render_geoms:
            self.viewer.add_geom(geom)
        results = []
        if flag == 0:
            self.viewer.set_bounds(-15, +15, -8, +8)
        else:
            # 计算无人车的位置范围并限制在范围内
            min_x = pos_copy[flag - 1][0] - detect_range
            max_x = pos_copy[flag - 1][0] + detect_range
            min_y = pos_copy[flag - 1][1] - detect_range
            max_y = pos_copy[flag - 1][1] + detect_range
            # 设置矩形视图边界，使其能够覆盖整个圆形范围
            self.viewer.set_bounds(min_x, max_x, min_y, max_y)

        for i, CAR in enumerate(self.CARs):  # 无人车以及攻击范围需要旋转
            if flag > 0 and (i != flag - 1) and (i not in all_opp) and (i not in all_nei):
                vel_copy[i][1] = 0
                vel_copy[i][0] = 0
            idx_ratio = self.length_temp1 // self.num_CARs
            for idx in range(idx_ratio):
                self.render_geoms_xform[idx_ratio * i + idx].set_translation(*pos_copy[i][0:2])

                if vel_copy[i][1] >= 0 and vel_copy[i][0] >= 0:
                    self.render_geoms_xform[idx_ratio * i + idx].set_rotation(
                        np.arctan(vel_copy[i][1] / vel_copy[i][0]))
                elif vel_copy[i][1] < 0 and vel_copy[i][0] >= 0:
                    self.render_geoms_xform[idx_ratio * i + idx].set_rotation(
                        np.arctan(vel_copy[i][1] / vel_copy[i][0]))
                else:
                    self.render_geoms_xform[idx_ratio * i + idx].set_rotation(
                        np.arctan(vel_copy[i][1] / vel_copy[i][0]) + np.pi)

        results.append(self.viewer.render(return_rgb_array=mode == 'rgb_array'))
        return results

    def render_static_obstacles(self):
        import rendering
        # 定义一些静态障碍物的矩形参数：位置（x, y）、宽度（w）、高度（h）
        obstacles = [
            {"pos": (1.9 - 10.85, 2.9 - 7.45), "width": 0.1, "height": 0.9, "color": (0, 0, 0)},
            {"pos": (1.9 - 10.85, 5 - 7.45), "width": 0.1, "height": 0.9, "color": (0, 0, 0)},
            {"pos": (1.9 - 10.85, 9.4 - 7.45), "width": 0.1, "height": 0.9, "color": (0, 0, 0)},
            {"pos": (1.9 - 10.85, 11.6 - 7.45), "width": 0.1, "height": 0.9, "color": (0, 0, 0)},

            {"pos": (5.9 - 10.85, 2.9 - 7.45), "width": 0.1, "height": 0.9, "color": (0, 0, 0)},
            {"pos": (5.9 - 10.85, 5 - 7.45), "width": 0.1, "height": 0.9, "color": (0, 0, 0)},
            {"pos": (5.9 - 10.85, 9.4 - 7.45), "width": 0.1, "height": 0.9, "color": (0, 0, 0)},
            {"pos": (5.9 - 10.85, 11.6 - 7.45), "width": 0.1, "height": 0.9, "color": (0, 0, 0)},

            {"pos": (2 - 10.85, 2.9 - 7.45), "width": 2.1, "height": 0.1, "color": (0, 0, 0)},
            {"pos": (2 - 10.85, 5.8 - 7.45), "width": 2.1, "height": 0.1, "color": (0, 0, 0)},
            {"pos": (2 - 10.85, 9.4 - 7.45), "width": 2.1, "height": 0.1, "color": (0, 0, 0)},
            {"pos": (2 - 10.85, 12.4 - 7.45), "width": 2.1, "height": 0.1, "color": (0, 0, 0)},

            {"pos": (5.4 - 10.85, 2.9 - 7.45), "width": 0.5, "height": 0.1, "color": (0, 0, 0)},
            {"pos": (5.4 - 10.85, 5.8 - 7.45), "width": 0.5, "height": 0.1, "color": (0, 0, 0)},
            {"pos": (5.4 - 10.85, 9.4 - 7.45), "width": 0.5, "height": 0.1, "color": (0, 0, 0)},
            {"pos": (5.4 - 10.85, 12.4 - 7.45), "width": 0.5, "height": 0.1, "color": (0, 0, 0)},

            {"pos": (15.6 - 10.85, 1.9 - 7.45), "width": 0.1, "height": 1.6, "color": (0, 0, 0)},
            {"pos": (15.6 - 10.85, 4.8 - 7.45), "width": 0.1, "height": 1.6, "color": (0, 0, 0)},
            {"pos": (15.6 - 10.85, 8.3 - 7.45), "width": 0.1, "height": 1.7, "color": (0, 0, 0)},
            {"pos": (15.6 - 10.85, 11.3 - 7.45), "width": 0.1, "height": 1.7, "color": (0, 0, 0)},

            {"pos": (19.6 - 10.85, 1.9 - 7.45), "width": 0.1, "height": 1.6, "color": (0, 0, 0)},
            {"pos": (19.6 - 10.85, 4.8 - 7.45), "width": 0.1, "height": 1.6, "color": (0, 0, 0)},
            {"pos": (19.6 - 10.85, 8.3 - 7.45), "width": 0.1, "height": 1.7, "color": (0, 0, 0)},
            {"pos": (19.6 - 10.85, 11.3 - 7.45), "width": 0.1, "height": 1.7, "color": (0, 0, 0)},

            {"pos": (15.7 - 10.85, 1.9 - 7.45), "width": 1.2, "height": 0.1, "color": (0, 0, 0)},
            {"pos": (15.7 - 10.85, 6.3 - 7.45), "width": 1.2, "height": 0.1, "color": (0, 0, 0)},
            {"pos": (15.7 - 10.85, 8.3 - 7.45), "width": 1.2, "height": 0.1, "color": (0, 0, 0)},
            {"pos": (15.7 - 10.85, 12.9 - 7.45), "width": 1.2, "height": 0.1, "color": (0, 0, 0)},

            {"pos": (18.2 - 10.85, 1.9 - 7.45), "width": 1.4, "height": 0.1, "color": (0, 0, 0)},
            {"pos": (18.2 - 10.85, 6.3 - 7.45), "width": 1.4, "height": 0.1, "color": (0, 0, 0)},
            {"pos": (18.2 - 10.85, 8.3 - 7.45), "width": 1.4, "height": 0.1, "color": (0, 0, 0)},
            {"pos": (18.2 - 10.85, 12.9 - 7.45), "width": 1.4, "height": 0.1, "color": (0, 0, 0)},

            # {"pos": (9.7 - 10.85, 3.9 - 7.45), "width": 0.1, "height": 7.1, "color": (0, 0, 0)},
            {"pos": (9.7 - 10.85, 3.9 - 7.45), "width": 0.1, "height": 1.8, "color": (0, 0, 0)},
            {"pos": (9.7 - 10.85, 9.2 - 7.45), "width": 0.1, "height": 1.8, "color": (0, 0, 0)},
            {"pos": (9.8 - 10.85, 3.9 - 7.45), "width": 1.1, "height": 0.1, "color": (0, 0, 0)},
            {"pos": (9.8 - 10.85, 10.9 - 7.45), "width": 1.1, "height": 0.1, "color": (0, 0, 0)},

            # {"pos": (12.2 - 10.85, 3.9 - 7.45), "width": 1.3, "height": 0.1, "color": (0, 0, 0)},
            # {"pos": (12.2 - 10.85, 10.9 - 7.45), "width": 1.3, "height": 0.1, "color": (0, 0, 0)},
            {"pos": (12.8 - 10.85, 3.9 - 7.45), "width": 0.7, "height": 0.1, "color": (0, 0, 0)},
            {"pos": (12.8 - 10.85, 10.9 - 7.45), "width": 0.7, "height": 0.1, "color": (0, 0, 0)},

            # {"pos": (13.5 - 10.85, 3.9 - 7.45), "width": 0.1, "height": 2.1, "color": (0, 0, 0)},
            # {"pos": (13.5 - 10.85, 8.6 - 7.45), "width": 0.1, "height": 2.4, "color": (0, 0, 0)},
            {"pos": (13.5 - 10.85, 3.9 - 7.45), "width": 0.1, "height": 1.8, "color": (0, 0, 0)},
            {"pos": (13.5 - 10.85, 9.2 - 7.45), "width": 0.1, "height": 1.8, "color": (0, 0, 0)},

            # {"pos": (10.8 - 10.85, 5.9 - 7.45), "width": 0.1, "height": 2.8, "color": (0, 0, 0)},

            # {"pos": (10.9 - 10.85, 5.9 - 7.45), "width": 2.6, "height": 0.1, "color": (0, 0, 0)},
            # {"pos": (10.9 - 10.85, 8.6 - 7.45), "width": 2.6, "height": 0.1, "color": (0, 0, 0)},
        ]

        for obs in obstacles:
            xform = rendering.Transform()

            # 创建矩形障碍物
            rect = rendering.make_polygon([
                (0, 0),
                (obs["width"], 0),
                (obs["width"], obs["height"]),
                (0, obs["height"]),
            ])

            # 设置障碍物的颜色
            rect.set_color(*obs["color"])

            # 添加到渲染几何体列表
            rect.add_attr(xform)
            self.render_geoms.append(rect)
            self.render_geoms_xform.append(xform)

            # 设置障碍物的位置
            xform.set_translation(*obs["pos"])

    def close(self):
        pass
