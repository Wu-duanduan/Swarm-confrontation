import numpy as np
from math import cos, sin, tan, pi, atan2, acos
from random import random, uniform, randint
from gym import spaces
import pygame
import pyglet
from pyglet import image
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
                CAR.color = np.array([0, i / self.num_BCARs, 1])
                CAR.attack_range = args.attack_range_B
                CAR.attack_angle = args.attack_angle_BR
                self.BCARs.append(CAR)
            elif i < args.num_RCARs + args.num_BCARs:
                CAR.enemy = True
                CAR.color = np.array([1, (i - self.num_BCARs) / self.num_RCARs, 0])
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
        pos_copy[0] = [0, 0, 0]
        vel_copy[0] = [0, 0, 0]
        if self.viewer is None:
            self.viewer = rendering.Viewer(900, 480)
            pygame.init()
        # 每次渲染时清除旧的几何对象
        self.render_geoms = []
        self.render_geoms_xform = []
        # 初始化pygame用于文本渲染

        self.CARs = []
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

        self.length_temp1 = len(self.render_geoms)

        # 渲染静态障碍物
        self.render_static_obstacles()
        self.viewer.geoms = []
        for geom in self.render_geoms:
            self.viewer.add_geom(geom)
        results = []
        if flag == 0:
            self.viewer.set_bounds(-15, +15, -8, +8)
            # self.viewer.transform.set_rotation(np.pi / 2)
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
        self.viewer.light_source = (0, 0)
        self.viewer.draw_shadow()
        results.append(self.viewer.render(return_rgb_array=mode == 'rgb_array'))
        return results
    
    def vel2yaw(self, vel):
        if vel[1] >= 0 and vel[0] >= 0:
            return np.arctan(vel[1] / vel[0])
        elif vel[1] < 0 and vel[0] >= 0:
            return np.arctan(vel[1] / vel[0])
        else:
            return np.arctan(vel[1] / vel[0]) + np.pi

    def render_BEV(self, pos, vel, detect_range, FOV, flag, mode='rgb_array'):
        viewer_size = 500
        if FOV > np.pi:
            FOV = FOV / 180 * np.pi
        if self.viewer is None:
            self.viewer = rendering.Viewer(viewer_size, viewer_size)
            pygame.init()
        # 每次渲染时清除旧的几何对象
        self.render_geoms = []
        self.render_geoms_xform = []

        ego_pos = pos[flag - 1][:2]
        ego_vel = vel[flag - 1][:2]
        # print(f'ego_vel: {ego_vel}')
        
        ego_yaw = self.vel2yaw(ego_vel)
        # print(f'ego_yaw: {ego_yaw / np.pi * 180}')

        for i, CAR in enumerate(self.CARs):  # 添加无人车
            if i == flag - 1:
                CAR.color = np.array([0, 1, 0])
            xform = rendering.Transform()
            xform.set_translation(*pos[i][0:2])
            yaw = self.vel2yaw(vel[i][:2])
            xform.set_rotation(yaw)
            for x in rendering.make_CAR(CAR.size):
                x.set_color(*CAR.color, 0.5)
                x.add_attr(xform)
                self.render_geoms.append(x)
                self.render_geoms_xform.append(xform)

        self.length_temp1 = len(self.render_geoms)
        
        # 渲染静态障碍物
        self.render_static_obstacles(ego_pos, ego_yaw, BEV_mode=True)
        self.viewer.draw_shadow(self.length_temp1)

        self.viewer.geoms = []
        for geom in self.render_geoms:
            self.viewer.add_geom(geom)
        results = []
        if flag == 0:
            self.viewer.set_bounds(-15, +15, -8, +8)
        else:
            self.viewer.camera_follow(ego_pos, ego_yaw, FOV, detect_range)
            self.viewer.light_source = ego_pos
        
        results.append(self.viewer.render(return_rgb_array=mode == 'rgb_array'))

        color_buffer = pyglet.image.get_buffer_manager().get_color_buffer()
        # 获取图像的原始像素数据
        image_data = color_buffer.get_image_data()
        # 将数据转换为 NumPy 数组（需要转换为 RGB 格式）
        img_data = np.frombuffer(image_data.get_data('RGB', image_data.width * 3), dtype=np.uint8)
        img_data = img_data.reshape((image_data.height, image_data.width, 3))

        # 翻转图像的垂直方向
        img_data = np.flipud(img_data)  # 或者 img_data = img_data[::-1]
        # 判断是否存在特定颜色
        present_ids = []
        for car in self.CARs:
            tolerance = 5
            # 转换颜色到0-255的整数范围
            color_rgb = (car.color * 0.5 * 255 + np.array([1, 1, 1]) * 0.5 * 255).round().astype(np.int32)
            lower = np.maximum(color_rgb - tolerance, 0)
            upper = np.minimum(color_rgb + tolerance, 255)
            lower = lower.astype(np.uint8)
            upper = upper.astype(np.uint8)
            # 创建颜色掩膜
            mask = np.all((img_data >= lower) & (img_data <= upper), axis=-1)
            if np.any(mask):
                present_ids.append(car.id)
        print(present_ids)
        return results

    def render_static_obstacles(self, ego_pos=(0, 0), ego_yaw=0, BEV_mode=False):
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
