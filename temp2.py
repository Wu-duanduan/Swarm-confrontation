import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# 无人车模型：简单的二阶动力学模型
class Vehicle:
    def __init__(self, x, y, vx, vy, ax, ay, max_speed=2):
        self.x = x  # 位置
        self.y = y  # 位置
        self.vx = vx  # 速度
        self.vy = vy  # 速度
        self.ax = ax  # 加速度
        self.ay = ay  # 加速度
        self.max_speed = max_speed  # 最大速度

    def update(self, dt):
        # 更新速度和位置
        self.vx += self.ax * dt
        self.vy += self.ay * dt

        # 限制速度，确保不超过最大速度
        speed = np.sqrt(self.vx ** 2 + self.vy ** 2)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            self.vx *= scale
            self.vy *= scale

        self.x += self.vx * dt
        self.y += self.vy * dt


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


# 计算两台无人机之间的欧几里得距离
def distance(drone1, drone2):
    return np.sqrt((drone1.x - drone2.x) ** 2 + (drone1.y - drone2.y) ** 2)


# 目标函数：确保每辆小车都被至少一台无人机探测到，并考虑避撞约束
def objective(params, drones, vehicles, dt, prediction_horizon=5, min_distance=5):
    total_count = 0
    penalty = 0
    prediction_horizon = int(prediction_horizon)  # 确保 prediction_horizon 是整数
    for t in range(prediction_horizon):
        # 预测无人车状态
        for vehicle in vehicles:
            vehicle.update(dt)

        # 确保每辆车至少被一台无人机覆盖
        for vehicle in vehicles:
            covered = False
            for drone in drones:
                if in_detection_range(drone, vehicle):
                    covered = True
                    break
            if covered:
                total_count += 1

        # 移动无人机
        for i, drone in enumerate(drones):
            dx, dy = params[i * 2:i * 2 + 2]
            drone.move(dx, dy)

        # 添加避撞约束：确保无人机之间的距离大于 min_distance
        for i in range(len(drones)):
            for j in range(i + 1, len(drones)):
                dist = distance(drones[i], drones[j])
                if dist < min_distance:
                    penalty += (min_distance - dist) ** 2  # 惩罚违反约束的情况

    return -total_count + penalty  # 最小化目标函数（最大化每辆车被至少一台无人机覆盖的数量），加上惩罚


# 设置参数
num_vehicles = 5
num_drones = 3
detection_range = 10.0
dt = 0.1  # 时间步长
min_distance = 5.0  # 无人机之间的最小安全距离

# 初始化无人车
vehicles = [Vehicle(np.random.uniform(0, 50), np.random.uniform(0, 50),
                    np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                    np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), max_speed=1)
            for _ in range(num_vehicles)]

# 初始化无人机
drones = [Drone(x=np.random.uniform(0, 50), y=np.random.uniform(0, 50), detection_range=detection_range)
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
