########################################
# 用来为动力学约束提供工具函数
########################################
import numpy as np
import pygame


#################################################################################
# 计算障碍物的四个角的坐标，仅在初始化时调用一次，暂时不进行优化
#################################################################################
def calculate_obstacles_corner(obstacles_center, obstacles_radius = 0.1) -> list:
    '''
    从障碍物的中心点和半径, 计算出障碍物的四个角的坐标
    obstacles_center: 障碍物的中心点, shape=(n, 2)
    obstacles_radius: 障碍物的半径, shape=1
    return: 障碍物的四个角的坐标, shape=(n, 4, 2)
    '''
    if len(obstacles_center) == 0:
        return []
    obstacles_center = np.array(obstacles_center)
    return calculate_all_rectangles(obstacles_center[:, :2], obstacles_radius * 2)

def is_connected(center1, center2, side_length):
    """判断两个小障碍物是否连接"""
    x1, y1 = center1
    x2, y2 = center2
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    return (dx <= side_length and dy == 0) or (dy <= side_length and dx == 0)

def dfs(centers, index, visited, group, side_length):
    """深度优先搜索，将相互连接的小障碍物加入同一组"""
    visited[index] = True
    group.append(centers[index])
    for i in range(len(centers)):
        if not visited[i] and is_connected(centers[index], centers[i], side_length):
            dfs(centers, i, visited, group, side_length)
        
def group_connected_obstacles(centers, side_length):
    """将小障碍物分组"""
    visited = [False] * len(centers)
    groups = []
    for i in range(len(centers)):
        if not visited[i]:
            group = []
            dfs(centers, i, visited, group, side_length)
            groups.append(group)
    return groups

def calculate_rectangle_corners(group, side_length):
    """计算一组小障碍物组成的长方形的四角坐标"""
    group = np.array(group)
    x_coords = group[:, 0]
    y_coords = group[:, 1]
    min_x = np.min(x_coords)
    max_x = np.max(x_coords)
    min_y = np.min(y_coords)
    max_y = np.max(y_coords)
    half_side = side_length / 2
    bottom_left = (min_x - half_side, min_y - half_side)
    bottom_right = (max_x + half_side, min_y - half_side)
    top_right = (max_x + half_side, max_y + half_side)
    top_left = (min_x - half_side, max_y + half_side)
    return [bottom_left, bottom_right, top_right, top_left]

def calculate_all_rectangles(centers, side_length=0.1):
    """计算所有小障碍物分组后组成的长方形的四角坐标"""
    groups = group_connected_obstacles(centers, side_length)
    rectangles = []
    for group in groups:
        corners = calculate_rectangle_corners(group, side_length)
        rectangles.append(corners)
    return rectangles

#################################################################################
# 测试时对小车和障碍物进行渲染
#################################################################################
def test_case_render(qTotal, vTotal, timestep, cars_size,
                      screen_size=[800, 600], world_size=[20, 20], obstacles_corner=[],
                      test_case_name='"Car Movement Test"'):
    '''
    测试用例渲染
    todo: 区分车头朝向和速度方向
    qTotal: 位置, shape=(n, total_step, 3)
    vTotal: 速度, shape=(n, total_step, 2)
    timestep: 时间步长
    cars_size: 小车的长宽, shape=(n, 2)
    screen_size: 屏幕大小, shape=(2,)
    world_size: 世界大小, shape=(2,)
    obstacles_corner: 障碍物的四个角的坐标, shape=(n, 4, 2)
    test_case_name: 测试用例名称
    '''
    # 初始化pygame
    pygame.init()
    screen_width = screen_size[0]
    screen_height = screen_size[1]
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption(test_case_name)

    # 定义颜色
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)  # 用于表示速度方向的颜色
    BLACK = (0, 0, 0)  # 用于表示障碍物的颜色

    # 缩放因子，将物理坐标转换为屏幕坐标
    scale_x = screen_width / world_size[0]  # 假设物理世界的 x 范围是 -10 到 10
    scale_y = screen_height / world_size[1]  # 假设物理世界的 y 范围是 -10 到 10

    running = True
    step = 0
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(WHITE)

        # 渲染障碍物
        for obstacle in obstacles_corner:
            obstacle_points = []
            for point in obstacle:
                x = int(point[0] * scale_x + screen_width / 2)
                y = int(-point[1] * scale_y + screen_height / 2)
                obstacle_points.append((x, y))
            pygame.draw.polygon(screen, BLACK, obstacle_points)

        if step < len(qTotal):
            # 遍历所有小车
            for i in range(len(qTotal[step])):
                # 获取当前小车的位置和速度
                center = qTotal[step][i][:2]
                velocity = vTotal[step][i]
                size = cars_size[i]

                # 计算小车的四个角标
                half_width = size[0] / 2
                half_height = size[1] / 2

                # 这里简单假设小车的方向与速度方向一致
                angle = np.arctan2(velocity[1], velocity[0])
                rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

                corners = np.array([
                    [-half_width, -half_height],
                    [half_width, -half_height],
                    [half_width, half_height],
                    [-half_width, half_height]
                ])

                # 旋转角标
                rotated_corners = np.dot(corners, rotation_matrix.T)

                # 平移角标到中心点
                translated_corners = rotated_corners + center

                # 将物理坐标转换为屏幕坐标
                car_points = []
                for point in translated_corners:
                    x = int(point[0] * scale_x + screen_width / 2)
                    y = int(-point[1] * scale_y + screen_height / 2)
                    car_points.append((x, y))

                # 绘制小车
                pygame.draw.polygon(screen, RED, car_points)

                # 将小车中心转换为屏幕坐标
                center_screen_x = int(center[0] * scale_x + screen_width / 2)
                center_screen_y = int(-center[1] * scale_y + screen_height / 2)

                # 绘制速度方向线段
                vx, vy = vTotal[step][i]
                vx_screen = int(vx * scale_x)
                vy_screen = int(-vy * scale_y)  # 负号是因为 pygame 的 y 轴向下
                end_x = center_screen_x + vx_screen
                end_y = center_screen_y + vy_screen
                pygame.draw.line(screen, BLUE, (center_screen_x, center_screen_y), (end_x, end_y), 2)

            step += 1
        else:
            # 当遍历完qTotal后，将running标志设置为False，退出循环
            running = False

        pygame.display.flip()
        clock.tick(1 / timestep)

    pygame.quit()


#########################################################################
# 检测两个物体相撞后的位置和速度
#########################################################################
def collision_response(c1, c2, v1, v2, m1, m2, collision_coefficient=0.8, timestep=0.01):
    '''
    检测两个物体相撞后的位置和速度，物体均是轴对称的（长方形）
    c1: 物体1的四个角标, shape=(4,2)
    c2: 物体2的四个角标, shape=(4,2)
    v1: 物体1的速度, shape=(2,)
    v2: 物体2的速度, shape=(2,)
    m1: 物体1的质量, shape=1
    m2: 物体2的质量, shape=1
    collision_coefficient: 碰撞系数, shape=1
    timestep: 时间步长, shape=1
    return: 物体1和物体2相撞后的位置和速度
    '''
    # 提取包围盒坐标
    min_x1, max_x1 = np.min(c1[:, 0]), np.max(c1[:, 0])
    min_y1, max_y1 = np.min(c1[:, 1]), np.max(c1[:, 1])
    min_x2, max_x2 = np.min(c2[:, 0]), np.max(c2[:, 0])
    min_y2, max_y2 = np.min(c2[:, 1]), np.max(c2[:, 1])

    # 碰撞检测
    collision = (max_x1 >= min_x2) and (min_x1 <= max_x2) and \
                (max_y1 >= min_y2) and (min_y1 <= max_y2)
    
    if not collision:
        return c1, c2, v1, v2

    # 计算重叠量
    overlap_x = min(max_x1, max_x2) - max(min_x1, min_x2)
    overlap_y = min(max_y1, max_y2) - max(min_y1, min_y2)

    # 确定碰撞法线方向
    if overlap_x < overlap_y:
        # X轴方向碰撞
        center1 = (min_x1 + max_x1) / 2
        center2 = (min_x2 + max_x2) / 2
        n = np.array([1, 0]) if center1 < center2 else np.array([-1, 0])
    else:
        # Y轴方向碰撞
        center1 = (min_y1 + max_y1) / 2
        center2 = (min_y2 + max_y2) / 2
        n = np.array([0, 1]) if center1 < center2 else np.array([0, -1])

    # 计算实际重叠量
    if n[0] != 0:  # X轴方向
        overlap = max_x1 - min_x2 if n[0] > 0 else max_x2 - min_x1
    else:          # Y轴方向
        overlap = max_y1 - min_y2 if n[1] > 0 else max_y2 - min_y1

    # 计算法线方向的相对速度
    v_rel = np.dot(v1 - v2, n)
    
    # 分离状态不处理
    if v_rel > 0:
        return c1, c2, v1, v2

    # 计算冲量
    j = -(1 + collision_coefficient) * v_rel / (1/m1 + 1/m2)
    
    # 更新速度
    v1_new = v1 + (j * n) / m1
    v2_new = v2 - (j * n) / m2

    # 计算位置修正量
    total_mass = m1 + m2
    delta1 = -(m2 / total_mass) * overlap * n if m2 != np.inf else -overlap * n
    delta2 = (m1 / total_mass) * overlap * n

    # 应用位置修正
    new_c1 = c1 + delta1
    new_c2 = c2 + delta2

    print("###################################################")
    print("发生碰撞！")
    print("###################################################")

    return new_c1, new_c2, v1_new, v2_new

#############################################################################################
# 碰撞测试
#############################################################################################
def test_collision_response(q1, q2, v1, v2, m1, m2, collision_coefficient=0.8, friction_coefficient=0.8, timestep=0.01, totaltime=1, test_model="GUI"):
    import pybullet as p
    import pybullet_data
    import time

    # 初始化 PyBullet 环境
    if test_model == "GUI":
        physicsClient = p.connect(p.GUI)  # 连接到 GUI 模式
    else:
        physicsClient = p.connect(p.DIRECT)  # 连接到 DIRECT 模式
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 设置数据路径
    p.setGravity(0, 0, -9.81)  # 设置重力

    # 加载地面模型
    planeId = p.loadURDF("plane.urdf")

    # 已知参数
    # 小车 1
    position1 = q1  # 位置
    orientation1 = velocity_to_orientation(v1)  # 车头朝向，初始为 0 度
    velocity1 = v1  # 速度
    mass1 = m1  # 质量
    restitution1 = collision_coefficient  # 碰撞系数
    friction1 = friction_coefficient  # 与地面的摩擦系数

    # 小车 2
    position2 = q2  # 位置
    orientation2 = velocity_to_orientation(v2)
    velocity2 = v2  # 速度
    mass2 = m2  # 质量
    restitution2 = collision_coefficient # 碰撞系数
    friction2 = friction_coefficient  # 与地面的摩擦系数

    # 创建小车模型
    car1 = p.loadURDF("racecar/racecar.urdf", position1, orientation1)
    car2 = p.loadURDF("racecar/racecar.urdf", position2, orientation2)

    # 设置小车质量
    p.changeDynamics(car1, -1, mass=mass1)
    p.changeDynamics(car2, -1, mass=mass2)

    # 设置小车碰撞系数
    p.changeDynamics(car1, -1, restitution=restitution1)
    p.changeDynamics(car2, -1, restitution=restitution2)

    # 设置小车与地面的摩擦系数
    p.changeDynamics(car1, -1, lateralFriction=friction1)
    p.changeDynamics(car2, -1, lateralFriction=friction2)

    # 设置小车初始速度
    p.resetBaseVelocity(car1, linearVelocity=velocity1)
    p.resetBaseVelocity(car2, linearVelocity=velocity2)

    # 模拟时间和时间步长
    simulation_time = totaltime  # 模拟时间（秒）
    time_step = timestep  # 时间步长（秒）
    num_steps = int(simulation_time / time_step)

    # 存储结果
    results = []

    # 模拟碰撞过程
    for _ in range(num_steps):
        # 获取小车的位置、车头朝向和速度
        pos1, orn1 = p.getBasePositionAndOrientation(car1)
        pos2, orn2 = p.getBasePositionAndOrientation(car2)
        lin_vel1, ang_vel1 = p.getBaseVelocity(car1)
        lin_vel2, ang_vel2 = p.getBaseVelocity(car2)

        # 记录结果
        results.append((pos1, orn1, lin_vel1, pos2, orn2, lin_vel2))

        # 进行一步模拟
        p.stepSimulation()
        if test_model == "GUI":
            time.sleep(time_step)

    # 断开连接
    p.disconnect()

    return results

def velocity_to_orientation(velocity):
    import pybullet as p
    # 归一化速度向量
    velocity = np.array(velocity)
    norm = np.linalg.norm(velocity)
    if norm == 0:
        # 如果速度为零，返回默认朝向（无旋转）
        return p.getQuaternionFromEuler([0, 0, 0])
    velocity_normalized = velocity / norm

    # 定义参考方向（这里假设为 x 轴正方向）
    reference_direction = np.array([1, 0, 0])

    # 计算旋转轴
    rotation_axis = np.cross(reference_direction, velocity_normalized)
    rotation_axis_norm = np.linalg.norm(rotation_axis)
    if rotation_axis_norm == 0:
        # 如果速度方向与参考方向相同或相反
        if np.dot(reference_direction, velocity_normalized) > 0:
            # 方向相同，无旋转
            return p.getQuaternionFromEuler([0, 0, 0])
        else:
            # 方向相反，绕 z 轴旋转 180 度
            return p.getQuaternionFromEuler([0, 0, np.pi])
    rotation_axis = rotation_axis / rotation_axis_norm

    # 计算旋转角度
    rotation_angle = np.arccos(np.dot(reference_direction, velocity_normalized))

    # 将旋转轴和旋转角度转换为四元数
    return p.getQuaternionFromAxisAngle(rotation_axis, rotation_angle)

def quaternion_to_direction(quaternion):
    import pybullet as p
    # 定义参考向量，这里选择 x 轴正方向的单位向量
    reference_vector = [1, 0, 0]
    # 使用 PyBullet 的 rotateVector 函数将参考向量按照四元数进行旋转
    rotated_vector = p.rotateVector(quaternion, reference_vector)
    return rotated_vector