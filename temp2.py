import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import copy

# 无人车模型：简单的二阶动力学模型
class Vehicle:
    def __init__(self, x, y, vx, vy, ax, ay, max_speed=2, boundary=(10, 40), max_acceleration=0.5):
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

    def in_detection_range(self, vehicle):
        # 判断小车是否在无人机的探测范围内
        return (self.x - vehicle.x)**2 + (self.y - vehicle.y)**2 <= self.detection_range**2

# 计算无人车是否在无人机的探测范围内
# def in_detection_range(drone, vehicle):
#     # 假设探测范围是矩形，范围在无人机下方
#     return (drone.x - drone.detection_range / 2 <= vehicle.x <= drone.x + drone.detection_range / 2 and
#             drone.y - drone.detection_range / 2 <= vehicle.y <= drone.y + drone.detection_range / 2)


# 计算两台无人机之间的欧几里得距离
def distance(drone1, drone2):
    return np.sqrt((drone1.x - drone2.x) ** 2 + (drone1.y - drone2.y) ** 2)


# 目标函数：确保每辆小车都被至少一台无人机探测到，并考虑避撞约束
def objective(params, drones, vehicles, dt, prediction_horizon=5, min_distance=5, move_penalty_factor=1.0):
    total_count = 0
    penalty = 0
    move_penalty = 0
    prediction_horizon = int(prediction_horizon)  # 确保 prediction_horizon 是整数

    # 为每个时间步更新无人车的状态
    for t in range(prediction_horizon):
        # 复制无人机对象进行预测
        drones_copy = [copy.deepcopy(drone) for drone in drones]

        # 预测无人车状态
        for vehicle in vehicles:
            vehicle.update(dt)

        # 预测无人机状态，并根据控制输入更新位置
        for i, drone in enumerate(drones_copy):
            dx, dy = params[i * 2:i * 2 + 2]
            drone.move(dx, dy)

        # 确保每辆车至少被一台无人机覆盖
        for vehicle in vehicles:
            covered = False
            for drone in drones_copy:
                if drone.in_detection_range(vehicle):
                    covered = True
                    break
            if covered:
                total_count += 1

        # 添加避撞约束：确保无人机之间的距离大于 min_distance
        for i in range(len(drones_copy)):
            for j in range(i + 1, len(drones_copy)):
                dist = np.sqrt((drones_copy[i].x - drones_copy[j].x) ** 2 + (drones_copy[i].y - drones_copy[j].y) ** 2)
                if dist < min_distance:
                    penalty += (min_distance - dist) ** 2  # 惩罚违反约束的情况

        # 添加移动惩罚：如果无人机的控制输入接近零，增加惩罚
        for i in range(0, len(params), 2):
            dx, dy = params[i], params[i + 1]
            move_penalty += move_penalty_factor * (dx ** 2 + dy ** 2)

    return -total_count + penalty + move_penalty  # 最小化目标函数（最大化每辆车被至少一台无人机覆盖的数量），加上惩罚


# 设置参数
num_vehicles = 5
num_drones = 3
detection_range = 10.0
dt = 0.1  # 时间步长
min_distance = 2  # 无人机之间的最小安全距离

# 初始化无人车
vehicles = [Vehicle(np.random.uniform(10, 40), np.random.uniform(10, 40),
                    0, 0,
                    0, 0, max_speed=1)
            for _ in range(num_vehicles)]

# 初始化无人机
drones = [Drone(x=np.random.uniform(10, 40), y=np.random.uniform(10, 40), detection_range=detection_range)
          for _ in range(num_drones)]

# 模拟路径规划
num_iterations = 100
trajectory = [[] for _ in range(num_drones)]  # 每个无人机的轨迹
for _ in range(num_iterations):
    # 使用优化方法来找到无人机的最优移动方向（MPC）
    initial_params = np.zeros(num_drones * 2)  # 每个无人机有两个控制输入 dx, dy
    result = minimize(objective, initial_params, args=(drones, vehicles, dt, min_distance),
                      bounds=[(-1, 1), (-1, 1)] * num_drones, options={'disp': True})

    # 记录每个无人机的位置
    for i, drone in enumerate(drones):
        dx, dy = result.x[i * 2:i * 2 + 2]
        drone.move(dx, dy)
        trajectory[i].append((drone.x, drone.y))

    # 更新无人车的位置
    for vehicle in vehicles:
        vehicle.update(dt)
        print(vehicle.x)

# 可视化结果
plt.figure(figsize=(10, 8))
for i, drone in enumerate(drones):
    trajectory_i = np.array(trajectory[i])
    plt.plot(trajectory_i[:, 0], trajectory_i[:, 1], label=f'Drone {i + 1} Path')

for vehicle in vehicles:
    plt.scatter(vehicle.x, vehicle.y, c='red', label='Vehicle' if vehicles.index(vehicle) == 0 else "")

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Multi-Drone Path Planning to Cover All Vehicles with Collision Avoidance')
plt.show()
