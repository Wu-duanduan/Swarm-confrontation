import copy

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# 无人车模型：简单的二阶动力学模型
class Vehicle:
    def __init__(self, x, y, vx, vy, ax, ay, max_speed=2, boundary=(15, 35), max_acceleration=0.5):
        self.x = x  # 位置
        self.y = y  # 位置
        self.vx = vx  # 速度
        self.vy = vy  # 速度
        self.ax = ax  # 加速度
        self.ay = ay  # 加速度
        self.max_speed = max_speed  # 最大速度
        self.boundary = boundary  # 运动边界（默认是 (0, 50) 的正方形区域）
        self.max_acceleration = max_acceleration  # 最大加速度

    def update(self, dt):
        # 随机生成加速度方向和大小
        random_ax = np.random.uniform(-self.max_acceleration, self.max_acceleration)
        random_ay = np.random.uniform(-self.max_acceleration, self.max_acceleration)

        # 更新加速度
        self.ax = random_ax
        self.ay = random_ay

        # 更新速度
        self.vx += self.ax * dt
        self.vy += self.ay * dt
        # 限制速度，确保不超过最大速度
        speed = np.sqrt(self.vx ** 2 + self.vy ** 2)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            self.vx *= scale
            self.vy *= scale

        # 更新位置
        self.x += self.vx * dt
        self.y += self.vy * dt

        # 限制位置在边界内
        self.x = np.clip(self.x, self.boundary[0], self.boundary[1])
        self.y = np.clip(self.y, self.boundary[0], self.boundary[1])

        # 如果碰到边界，可以设置反弹
        if self.x == self.boundary[0] or self.x == self.boundary[1]:
            self.vx = -self.vx  # 反弹
        if self.y == self.boundary[0] or self.y == self.boundary[1]:
            self.vy = -self.vy  # 反弹


# 无人机模型：简单的空中轨迹
class Drone:
    def __init__(self, x, y, detection_range):
        self.x = x  # 位置
        self.y = y  # 位置
        self.detection_range = detection_range  # 探测范围（矩形区域）

    def move(self, dx, dy):
        self.x += dx
        self.y += dy


# 计算无人车是否在无人机的探测范围内
def in_detection_range(drone, vehicle):
    # 假设探测范围是矩形，范围在无人机下方
    return (drone.x - drone.detection_range / 2 <= vehicle.x <= drone.x + drone.detection_range / 2 and
            drone.y - drone.detection_range / 2 <= vehicle.y <= drone.y + drone.detection_range / 2)

def objective(params, drone, vehicles, dt, prediction_horizon=5):
    drone_copy = copy.deepcopy(drone)
    dx, dy = params  # 解包 dx, dy
    total_count = 0
    for t in range(prediction_horizon):
        # 预测无人车状态
        for vehicle in vehicles:
            vehicle.update(dt)
            if in_detection_range(drone_copy, vehicle):
                total_count += 1
        # 移动无人机
        drone_copy.move(dx, dy)
    # 增加一个惩罚项，鼓励无人机移动
    penalty = 0.1 * (abs(dx) + abs(dy))  # 惩罚无人机不移动
    return -total_count + penalty  # 最小化目标函数（最大化覆盖车辆数），同时惩罚不移动


# 设置参数
num_vehicles = 5
detection_range = 10.0
dt = 0.1  # 时间步长

# 初始化无人车
vehicles = [Vehicle(np.random.uniform(15, 35), np.random.uniform(15, 35),
                    0, 0,
                    0, 0, max_speed=1)
            for _ in range(num_vehicles)]

# 初始化无人机
drone = Drone(x=25, y=25, detection_range=detection_range)

# 模拟路径规划
num_iterations = 100
trajectory = [(drone.x, drone.y)]
for _ in range(num_iterations):
    # 使用优化方法来找到无人机的最优移动方向（MPC）
    result = minimize(objective, [0, 0], args=(drone, vehicles, dt), bounds=[(-1, 1), (-1, 1)])

    # 更新无人机位置
    dx, dy = result.x
    drone.move(dx, dy)
    # 记录轨迹
    trajectory.append((drone.x, drone.y))
    # 更新无人车的位置
    for vehicle in vehicles:
        vehicle.update(dt)
        # print(vehicle.x)
# 可视化结果
trajectory = np.array(trajectory)
plt.plot(trajectory[:, 0], trajectory[:, 1], label='Drone Path')
for vehicle in vehicles:
    plt.scatter(vehicle.x, vehicle.y, c='red', label='Vehicle' if vehicles.index(vehicle) == 0 else "")
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Drone Path Planning to Cover Vehicles Using MPC (Slower Vehicles)')
plt.show()
