import numpy as np
import random
from copy import deepcopy

from physical_law_tool import calculate_obstacles_corner, test_case_render, collision_response, resolve_overlaps

from physical_law_tool import calculate_deceleration_motion, calculate_acceleration_motion_vector, calculate_circular_motion_vector

from physical_law_tool import resolve_overlaps_vector

class PhysicalLaw:
    # 不变的常量
    GRAVITY = 10 # 重力加速度
    BINS = 10 # 分箱数量

    def __init__(self,
                 cars_mass, cars_force, cars_power, cars_friction_coefficient, cars_size, cars_wheel_spacing, cars_wheel_radius,
                 obstacles_center, obstacles_radius,
                 timestep,
                 collision_coefficient = 0.001,
                 ):
        '''
        cars_mass: 小车质量, shape=(n,)
        cars_force: 小车的最大受力, shape=(n,)
        cars_power: 小车功率, shape=(n,)
        cars_friction_coefficient: 小车的摩擦系数, shape=(n,)
        cars_size: 小车的长宽, shape=(n, 2)
        cars_wheel_spacing: 小车的轮间距, shape=(n,)
        obstacles_corner: 障碍物的四个角的坐标, shape=(n, 4, 2)
        timestep: 时间步长
        collision_coefficient: 碰撞系数
        '''
        self.cars_mass = np.array(cars_mass)
        self.cars_force = np.array(cars_force)
        self.cars_power = np.array(cars_power)
        self.cars_friction_coefficient = np.array(cars_friction_coefficient)
        self.cars_size = np.array(cars_size)
        self.cars_wheel_spacing = np.array(cars_wheel_spacing)
        self.cars_wheel_radius = np.array(cars_wheel_radius)
        self.obstacles_corner = np.array(calculate_obstacles_corner(obstacles_center, obstacles_radius))
        self.timestep = timestep
        self.collision_coefficient = collision_coefficient
        
        # 车的最大受力
        self.cars_friction_force_max = self.cars_friction_coefficient * self.cars_mass * self.GRAVITY
        self.cars_force = np.minimum(self.cars_force, self.cars_friction_force_max)
        # 车的阻力（滚动摩擦力）
        self.cars_friction_force_rolling = self.cars_friction_coefficient / self.cars_wheel_radius * self.cars_mass * self.GRAVITY
    
    def get_qvNext(self, q, v, vNext) -> list | list:
        '''
        从位置q, 速度v, 下一时刻的目标速度vNext, 以及小车自身属性和障碍物设置, 计算下一时刻的位置qNext
        todo: 用numpy加速
        未考虑因素：
        1. 车的转动惯性
        2. 车的电机扭矩
        q: 位置, z轴数据固定, shape=(n, 3)
        v: 速度, shape=(n, 2)
        vNext: 下一时刻的目标速度, shape=(n, 2)
        return: 下一时刻的位置qNext, shape=(n, 3)
        return: 下一时刻的速度vNext, shape=(n, 2)
        '''
        q = np.array(deepcopy(q), dtype=np.float64)
        v = np.array(deepcopy(v), dtype=np.float64)
        vNext = np.array(vNext)
        mask_v = np.linalg.norm(v, axis=1) < 1e-10
        mask_vNext = np.linalg.norm(vNext, axis=1) < 1e-10
        v[mask_v] = vNext[mask_v] * 1e-10
        vNext[mask_vNext] = v[mask_vNext] * 1e-10
        mask = mask_v & mask_vNext
        v_ones = np.ones(v.shape)
        v_ones[:, 2] = 0
        v[mask] = v_ones[mask] * 1e-10
        vNext[mask] =  v_ones[mask] * 1e-10
        qActual, vActual = self.get_qvNext_micro(q, v[:, :2], vNext[:, :2])
        # # 如果aAcural的模长为0，则将v乘1e-10赋值给vActual
        # v[:, :2] = vActual
        # return qActual, v
        qActual, vActual = self.get_qvNext_formula(q, v[:, :2], vNext[:, :2])
        q[:, :2] = qActual
        v[:, :2] = vActual
        return q, v

    def get_qvNext_micro(self, q, v, vNext) -> list | list:
        '''
        从位置q, 速度v, 下一时刻的目标速度vNext, 以及小车自身属性和障碍物设置, 使用微元法计算下一时刻的位置qNext
        q: 位置, z轴数据固定, shape=(n, 3)
        v: 速度, shape=(n, 2)
        vNext: 下一时刻的目标速度, shape=(n, 2)
        return: 下一时刻的位置qNext, shape=(n, 3)
        return: 下一时刻的速度vNext, shape=(n, 2)
        '''
        micro_timestep = self.timestep / self.BINS
        micro_timestep_squared = micro_timestep ** 2
        mass_expanded = self.cars_mass[:, np.newaxis]

        # 预分配内存
        q_2d = q[:, :2].copy()
        v_current = v.copy()

        for i in range(self.BINS):
            # 计算小车需要的力的大小和方向
            # force_need = self.calculate_force_need(v, vNext)
            force_need = (vNext - v) / (self.timestep - i * micro_timestep) * mass_expanded
            # 计算小车实际受力
            force = self.calculate_force_actural(force_need, v)
            # 计算小车的加速度
            a = force / mass_expanded
            # 利用加速度计算小车位置和速度
            q_2d += v_current * micro_timestep + 0.5 * a * micro_timestep_squared
            v_current += a * micro_timestep

            # 碰撞检测
            q_2d, v_current = self.check_collisions(q_2d, v_current)

        q[:, :2] = q_2d
        return q, v_current

    def get_qvNext_formula(self, q, v, vNext) -> list | list:
        '''
        从位置q, 速度v, 下一时刻的目标速度vNext, 以及小车自身属性和障碍物设置, 使用公式法计算下一时刻的位置qNext
        q: 位置, z轴数据固定, shape=(n, 3)
        v: 速度, shape=(n, 2)
        vNext: 下一时刻的目标速度, shape=(n, 2)
        return: 下一时刻的位置qNext, shape=(n, 3)
        return: 下一时刻的速度vNext, shape=(n, 2)
        '''
        t_last = np.ones(self.cars_mass.shape) * self.timestep
        q_2d = q[:, :2].copy()
        v_2d = v.copy()

        # 判断需要加速还是减速
        accelerate_condition = np.linalg.norm(vNext, axis=1) > np.linalg.norm(v, axis=1)
        # 如果需要加速则分配全部时间给转向
        t_turn = t_last * accelerate_condition
        # 计算能提供的转向加速度
        a = self.cars_friction_force_max / self.cars_mass
        # 转向
        v_2d, q_2d, t = calculate_circular_motion_vector(q_2d, v_2d, vNext, a, t_turn)
        t_last -= t
        # 加速
        v_2d, q_2d, t = calculate_acceleration_motion_vector(self.cars_mass, q_2d, v_2d, vNext, self.cars_power, self.cars_friction_force_max, self.cars_friction_force_rolling, t_last)
        t_last -= t

        # 判断是否需要减速
        decelerate_condition = np.linalg.norm(vNext, axis=1) < np.linalg.norm(v, axis=1)
        # 如果需要减速则分配全部时间给减速
        t_turn = t_last * decelerate_condition
        # 减速
        v_2d, q_2d, t = calculate_deceleration_motion(self.cars_mass, q_2d, v_2d, vNext, self.cars_friction_force_max, self.cars_friction_force_rolling, t_turn)
        t_last -= t

        # 全体转向
        v_2d, q_2d, t = calculate_circular_motion_vector(q_2d, v_2d, vNext, a, t_last)
        t_last -= t
        # 剩余时间匀速运动
        q_2d += v_2d * t_last[:, np.newaxis]

        # 碰撞检测
        q_2d, v_2d = self.check_collisions(q_2d, v_2d)

        return q_2d, v_2d
        
    def get_cars_corners(self, centers, velocities):
        """
        根据所有小车的中心点位置、长宽和速度方向，计算所有小车的四个角标（向量化版本）
        :param centers: 所有小车的中心点位置，形状为 (m, 2)
        :param velocities: 所有小车的速度，形状为 (m, 2)
        :return: 所有小车的四个角标，形状为 (m, 4, 2)
        """
        m = centers.shape[0]
        car_sizes = self.cars_size  # 假设 self.cars_size 形状为 (m, 2)
        
        # 计算半宽和半高
        half_widths = car_sizes[:, 0] / 2  # (m,)
        half_heights = car_sizes[:, 1] / 2  # (m,)
        
        # 生成基础角点模板（未缩放）
        base_corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float32)  # (4, 2)
        
        # 通过广播生成所有小车的角点（缩放后）
        corners = np.empty((m, 4, 2))
        corners[:, :, 0] = half_widths[:, np.newaxis] * base_corners[:, 0]  # x分量
        corners[:, :, 1] = half_heights[:, np.newaxis] * base_corners[:, 1]  # y分量
        
        
        # 计算旋转角度
        angles = np.arctan2(velocities[:, 1], velocities[:, 0])  # (m,)

        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)
        
        # 构建旋转矩阵数组 (m, 2, 2)
        rotation_matrices = np.empty((m, 2, 2))
        rotation_matrices[:, 0, 0] = cos_angles
        rotation_matrices[:, 0, 1] = -sin_angles
        rotation_matrices[:, 1, 0] = sin_angles
        rotation_matrices[:, 1, 1] = cos_angles
        
        # 应用旋转：等价于 corners[i] @ rotation_matrices[i].T
        rotated_corners = np.einsum('...ij,...kj->...ik', corners, rotation_matrices)
        
        # 平移角点到中心点
        cars_corners = rotated_corners + centers[:, np.newaxis, :]
        
        return cars_corners

    def check_collisions(self, car_centers, car_velocities, collision_time=0.01):
        """
        检查小车与障碍物以及小车与小车之间是否发生碰撞，并计算碰撞后小车的位置和速度
        :param obstacles: 所有障碍物的四个角标，形状为 (n, 4, 2)
        :param car_centers: 所有小车的中心点位置，形状为 (m, 2)
        :param car_velocities: 所有小车的速度，形状为 (m, 2)
        :param car_sizes: 所有小车的长宽，形状为 (m, 2)
        :param collision_time: 碰撞时间
        :return: 一个包含四个元素的元组，分别为：
                 1. 碰撞后所有小车的位置，形状为 (m, 2)
                 2. 碰撞后所有小车的速度，形状为 (m, 2)
        """
        # return self.check_collisions_vectorized(car_centers, car_velocities, collision_time)
        # obstacles = self.obstacles_corner
        # m = len(car_centers)
        # n = len(obstacles)

        # 计算碰撞后小车的位置和速度
        new_car_centers = car_centers.copy()
        new_car_velocities = car_velocities.copy()

        # collision_happened = False

        # 检查小车与障碍物是否碰撞
        cars_corners = self.get_cars_corners(car_centers, car_velocities)
        # for i in range(m):
        #     car_corners = cars_corners[i]
        #     for j in range(n):
        #         new_c1, _, new_v1, _, collision = collision_detection(car_corners, obstacles[j], 
        #                                                   new_car_velocities[i], np.zeros_like(new_car_velocities[i]),
        #                                                   self.cars_mass[i], np.inf,
        #                                                   self.collision_coefficient, collision_time)
        #         # new_c1.shape = (4, 2)
        #         # new_car_centers[i].shape = (2,)
        #         new_car_centers[i] = np.average(new_c1, axis=0)
        #         new_car_velocities[i] = new_v1
        #         if collision:
        #             cars_corners = self.get_cars_corners(new_car_centers, new_car_velocities)
        #             car_corners = cars_corners[i]
        #             collision_happened = True

        # # 检查小车与小车之间是否碰撞
        # for i in range(m):
        #     car1_corners = cars_corners[i]
        #     for j in range(i + 1, m):
        #         car2_corners = cars_corners[j]
        #         new_c1, new_c2, new_v1, new_v2, collision = collision_detection(car1_corners, car2_corners,
        #                                                           new_car_velocities[i], new_car_velocities[j],
        #                                                           self.cars_mass[i], self.cars_mass[j],
        #                                                           self.collision_coefficient, collision_time)
        #         new_car_centers[i] = np.average(new_c1, axis=0)
        #         new_car_centers[j] = np.average(new_c2, axis=0)
        #         new_car_velocities[i] = new_v1
        #         new_car_velocities[j] = new_v2
        #         if collision:
        #             cars_corners = self.get_cars_corners(new_car_centers, new_car_velocities)
        #             car1_corners = cars_corners[i]
        #             collision_happened = True
        
        # 处理重叠的情况
        # if collision_happened:
        cars_corners, new_car_velocities = collision_response(cars_corners, new_car_velocities, self.cars_mass, self.cars_size, self.obstacles_corner, self.collision_coefficient, self.timestep / 10)
        cars_corners = resolve_overlaps_vector(cars_corners, self.obstacles_corner)
        new_car_centers = np.average(cars_corners, axis=1)
        return new_car_centers, new_car_velocities

    def calculate_trajectory_2d(self, q: np.array, v: np.array, a: np.array, m: int) -> tuple[np.array, np.array]:
        '''
        从位置q, 速度v, 加速度a, 以及小车自身属性和障碍物设置, 计算m个轨迹点
        q: 位置, z轴数据固定, shape=(n, 3)
        v: 速度, shape=(n, 2)
        a: 加速度, shape=(n, 2)
        m: 轨迹点的数量
        return: 轨迹点, shape=(n, m, 2)
        return: 对应速度, shape=(n, m, 2)
        '''
        # 生成 m 个等间距的时间点，范围从 0 到 self.timestep
        time_steps = np.linspace(0, self.timestep, m)
        # 扩展时间步的维度，使其可以与 q、v、a 进行广播操作
        time_steps = time_steps[:, np.newaxis, np.newaxis]

        # 只取 q 的前两维进行轨迹计算
        q_2d = q[:, :2]

        # 根据运动学公式计算轨迹点
        trajectory_points_2d = q_2d[np.newaxis, :, :] + v[np.newaxis, :, :] * time_steps + 0.5 * a[np.newaxis, :, :] * (time_steps ** 2)
        # 调整维度顺序，使其形状为 (n, m, 2)
        trajectory_points_2d = np.transpose(trajectory_points_2d, (1, 0, 2))

        # 根据运动学公式计算对应速度
        velocities = v[np.newaxis, :, :] + a[np.newaxis, :, :] * time_steps
        # 调整维度顺序，使其形状为 (n, m, 2)
        velocities = np.transpose(velocities, (1, 0, 2))

        return trajectory_points_2d, velocities

    def calculate_force_actural(self, force_need: np.array, v: np.array) -> np.array:
        '''
        根据摩擦系数计算小车实际受力，小车受力不超过摩擦力的最大值, 并考虑小车的功率限制
        force_need: 小车需要的力, shape=(n, 2)
        return: 小车的实际受力, shape=(n, 2)
        '''
        # 再根据功率限制计算小车的最大力，当小车速度为0时，力的大小为上限
        v_size = np.linalg.norm(v, axis=1)
        v_size = np.where(v_size < 1e-10, 1e-10, v_size)
        force_max = self.cars_power / v_size - self.cars_friction_force_rolling
        # 再根据摩擦系数计算小车实际受力，小车受力不超过动力的最大值
        force_actural_size = np.clip(force_max, 0, self.cars_force - self.cars_friction_force_rolling)
        # 同时实际受力也不能超过小车需要的力
        force_actural_size = np.clip(force_actural_size, 0, np.linalg.norm(force_need, axis=1))
        # 再计算需要的力的方向，如果需要的力的大小为0，则方向为0，防止除零错误
        force_need_norm = np.linalg.norm(force_need, axis=1, keepdims=True)
        # 避免除零错误，当力的大小接近0时，将其替换为一个极小的正数
        force_need_norm = np.where(force_need_norm < 1e-10, 1e10, force_need_norm)
        force_need_direction = force_need / force_need_norm

        # 再计算小车的实际受力
        force_actural = force_actural_size[:, np.newaxis] * force_need_direction
        return force_actural

def one_car_no_obstacles():
    '''
    单车无障碍物测试
    '''
    physical_law = PhysicalLaw(
        cars_mass=[100],  # 小车质量, 1维向量
        cars_force=[200],  # 小车动力, 1维向量
        cars_power=[100],  # 小车功率, 1维向量
        cars_friction_coefficient=[0.1],  # 小车的摩擦系数, 1维向量
        cars_size = [[0.5, 0.5]],  # 小车的长宽, 2维向量
        cars_wheel_spacing=[1],  # 小车的轮间距, 1维向量
        cars_wheel_radius=[2],  # 小车的轮半径, 1维向量
        obstacles_center=[],  # 障碍物的中心点, 2维向量
        obstacles_radius=0.1,  # 障碍物的半径, 1维向量
        timestep=0.1,  # 时间步长
        collision_coefficient=0.001,  # 碰撞系数
    )

    total_time = 5  # 总时间
    total_step = int(total_time / physical_law.timestep)  # 总步数
    q = [[0., 0., 0.]]  # 初始位置
    v = [[1e-10, 0.]]  # 初始速度

    # 目标速度
    vNexts = [[[random.random() * 10, random.random() * 10]] for _ in range(total_step)]

    # 开始测试
    qTotal = [q]
    vTotal = [v]
    for i in range(total_step):
        qActural, vActural = physical_law.get_qvNext(q, v, vNexts[i])
        qTotal.append(qActural)
        vTotal.append(vActural)
        q = qActural
        v = vActural
        print(f"第{i}步, vNext: {vNexts[i]}, vActural: {vActural}")

    # 渲染部分
    test_case_render(qTotal, vTotal, physical_law.timestep, physical_law.cars_size,
                     obstacles_corner=physical_law.obstacles_corner, 
                     test_case_name="One Car No Obstacles")

def one_car_with_obstacles():
    '''
    单车有障碍物测试
    '''
    physical_law = PhysicalLaw(
        cars_mass=[100],  # 小车质量, 1维向量
        cars_force=[200],  # 小车动力, 1维向量
        cars_power=[100],  # 小车功率, 1维向量
        cars_friction_coefficient=[0.1],  # 小车的摩擦系数, 1维向量
        cars_size = [[0.5, 0.5]],  # 小车的长宽, 2维向量
        cars_wheel_spacing=[1],  # 小车的轮间距, 1维向量
        cars_wheel_radius=[2],  # 小车的轮半径, 1维向量
        obstacles_center=[[1, 0], [1, 0.2], [1, -0.2]],  # 障碍物的中心点, 2维向量
        obstacles_radius=0.1,  # 障碍物的半径, 1维向量
        timestep=0.1,  # 时间步长
        collision_coefficient=0.001,  # 碰撞系数
    )

    total_time = 2  # 总时间
    total_step = int(total_time / physical_law.timestep)  # 总步数
    q = [[0., 0., 0.]]  # 初始位置
    v = [[1e-10, 0.]]  # 初始速度

    # 目标速度
    vNexts = [[[i * 0.1, i * 0.05]] for i in range(total_step)]

    # 开始测试
    qTotal = [q]
    vTotal = [v]
    for i in range(total_step):
        qActural, vActural = physical_law.get_qvNext(q, v, vNexts[i])
        qTotal.append(qActural)
        vTotal.append(vActural)
        q = qActural
        v = vActural
        # print(f"第{i}步, vNext: {vNexts[i]}, vActural: {vActural}")

    # 渲染部分
    test_case_render(qTotal, vTotal, physical_law.timestep, physical_law.cars_size,
                     obstacles_corner=physical_law.obstacles_corner, 
                     test_case_name="One Car With Obstacles")

def multi_cars_no_obstacles():
    '''
    多车无障碍物测试
    '''
    n = 3
    physical_law = PhysicalLaw(
        cars_mass=[100] * n,  # 小车质量, 1维向量
        cars_force=[200] * n,  # 小车动力, 1维向量
        cars_power=[100] * n,  # 小车功率, 1维向量
        cars_friction_coefficient=[0.1] * n,  # 小车的摩擦系数, 1维向量
        cars_size = [[0.5, 0.5]] * n,  # 小车的长宽, 2维向量
        cars_wheel_spacing=[1] * n,  # 小车的轮间距, 1维向量
        cars_wheel_radius=[2] * n,  # 小车的轮半径, 1维向量
        obstacles_center=[],  # 障碍物的中心点, 2维向量
        obstacles_radius=0.1,  # 障碍物的半径, 1维向量
        timestep=0.1,  # 时间步长
        collision_coefficient=1,  # 碰撞系数
    )

    total_time = 5  # 总时间
    total_step = int(total_time / physical_law.timestep)  # 总步数
    q = [[0., 0., 0.], [1., 0., 0.], [-5., 0., 0.]]  # 初始位置
    v = [[1e-10, 0.], [-1e-10, 0.], [1e-10, 0.]]  # 初始速度

    # 目标速度
    vNexts = [[[i * 0.1, i * 0.05], [-i * 0.1, i * 0.05], [0, 0]] for i in range(total_step)]

    # 开始测试
    qTotal = [q]
    vTotal = [v]
    for i in range(total_step):
        qActural, vActural = physical_law.get_qvNext(q, v, vNexts[i])
        qTotal.append(qActural)
        vTotal.append(vActural)
        q = qActural
        v = vActural
        print(f"第{i}步, vNext: {vNexts[i]}, vActural: {vActural}")

    # 渲染部分
    test_case_render(qTotal, vTotal, physical_law.timestep, physical_law.cars_size,
                     obstacles_corner=physical_law.obstacles_corner, 
                     test_case_name="One Car Without Obstacles")

def multi_cars_with_obstacles():
    '''
    多车有障碍物测试
    '''
    n = 3
    physical_law = PhysicalLaw(
        cars_mass=[2.5] * n,  # 小车质量, 1维向量
        cars_force=[17.5] * n,  # 小车动力, 1维向量
        cars_power=[45] * n,  # 小车功率, 1维向量
        cars_friction_coefficient=[0.3] * n,  # 小车的摩擦系数, 1维向量
        cars_size = [[0.3 * 2, 0.3 * 7 / 5]] * n,  # 小车的长宽, 2维向量
        cars_wheel_spacing=[1] * n,  # 小车的轮间距, 1维向量
        cars_wheel_radius=[2] * n,  # 小车的轮半径, 1维向量
        obstacles_center=[],  # 障碍物的中心点, 2维向量
        obstacles_radius=0.1,  # 障碍物的半径, 1维向量
        timestep=0.1,  # 时间步长
        collision_coefficient=0.95, # 碰撞后速度的衰减系数
    )

    total_time = 1.8  # 总时间
    total_step = int(total_time / physical_law.timestep)  # 总步数
    q = [[0., 0., 0.], [1., 0., 0.], [-5., 0., 0.]]  # 初始位置
    v = [[1e-10, 0., 0.], [-1e-10, 0., 0.], [1e-10, 0., 0.]]  # 初始速度

    # 目标速度
    vNexts = [[[i * 0.1, i * 0.05, 0.], [-i * 0.1, i * 0.05, 0.], [0, 0, 0.]] for i in range(total_step)]

    # 开始测试
    qTotal = [q]
    vTotal = [v]
    for i in range(total_step):
        qActural, vActural = physical_law.get_qvNext(q, v, vNexts[i])
        qTotal.append(qActural)
        vTotal.append(vActural)
        q = qActural
        v = vActural
        print(f"第{i}步, vNext: {vNexts[i]}, vActural: {vActural}, qActural: {qActural}")

    # 渲染部分
    test_case_render(qTotal, vTotal, physical_law.timestep, physical_law.cars_size,
                     obstacles_corner=physical_law.obstacles_corner, 
                     test_case_name="One Car Without Obstacles")

def collision_test():
    '''
    碰撞测试
    '''
    # 初始数据
    q1 = [-1, 0, 0]
    q2 = [0, 1, 0]
    v1 = [5, 0, 0]
    v2 = [0, -5, 0]
    m1 = 100
    m2 = 100
    collision_coefficient = 0.8
    friction_coefficient = 0.1
    totaltime = 2
    timestep=0.1
    timebins = 20

    from physical_law_tool import test_collision_response, quaternion_to_direction
    result_expect = test_collision_response(
        q1=q1,
        q2=q2,
        v1=v1,
        v2=v2,
        m1=m1,
        m2=m2,
        collision_coefficient=collision_coefficient,
        friction_coefficient=friction_coefficient,
        totaltime=totaltime,
        timestep=timestep/10,
        test_model="DIRECT"
        # videoLog=True,
    )
    n = 2
    physical_law = PhysicalLaw(
        cars_mass=[m1, m2],  # 小车质量, 1维向量
        cars_force=[200, 200],  # 小车动力, 1维向量
        cars_power=[100, 100],  # 小车功率, 1维向量
        cars_friction_coefficient=[friction_coefficient] * n,  # 小车的摩擦系数, 1维向量
        cars_size = [[0.5, 0.5]] * n,  # 小车的长宽, 2维向量
        cars_wheel_spacing=[1] * n,  # 小车的轮间距, 1维向量
        cars_wheel_radius=[1] * n,  # 小车的轮半径, 1维向量
        obstacles_center=[],  # 障碍物的中心点, 2维向量
        obstacles_radius=0.1,  # 障碍物的半径, 1维向量
        timestep=timestep,  # 时间步长
        collision_coefficient=collision_coefficient,  # 碰撞系数
    )

    vNext1 = [5, 0, 0]
    vNext2 = [0, -5, 0]
    def round_tuple(t):
        if isinstance(t, (tuple, list)):
            return tuple(round(x, 2) if isinstance(x, (int, float)) else x for x in t)
        return t
    for i in range(timebins):
        q1_expect, o1_expect, v1_expect, q2_expect, o2_expect, v2_expect = result_expect[i*10]
        print(f"q1_expect: {round_tuple(q1_expect)}, \nv1_expect: {round_tuple(v1_expect)}, \nq2_expect: {round_tuple(q2_expect)}, \nv2_expect: {round_tuple(v2_expect)}")
        (q1_actual, q2_actual), (v1_actual, v2_actual) = physical_law.get_qvNext(
            q=[q1, q2],
            v=[v1, v2],
            vNext=[vNext1, vNext2],
        )
        print(f"q1_actual: {q1_actual.round(2)}, \nv1_actual: {v1_actual.round(2)}, \nq2_actual: {q2_actual.round(2)}, \nv2_actual: {v2_actual.round(2)}")
        q1 = q1_actual
        v1 = v1_actual
        q2 = q2_actual
        v2 = v2_actual
        print("-" * 100)


    # for q1, o1, v1, q2, o2, v2 in result:
    #     print(f"o1: {quaternion_to_direction(o1)[:2]}, \nv1: {v1[:2]}, \no2: {quaternion_to_direction(o2)[:2]}, \nv2: {v2[:2]}")
    

if __name__ == '__main__':
    random.seed(1)
    # one_car_no_obstacles()
    # one_car_with_obstacles()
    # multi_cars_no_obstacles()
    multi_cars_with_obstacles()
    # collision_test()