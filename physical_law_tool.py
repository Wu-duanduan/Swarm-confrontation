########################################
# 用来为动力学约束提供工具函数
########################################
import numpy as np
import pygame
import timeit


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

def calculate_rectangle_size(rectangle):
    """根据长方形的四个角标计算长方形的大小"""
    x_coords = [corner[0] for corner in rectangle]
    y_coords = [corner[1] for corner in rectangle]
    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)
    return width, height

#################################################################################
# 测试时对小车和障碍物进行渲染
#################################################################################
def test_case_render(qTotal, vTotal, timestep, cars_size,
                      screen_size=[800, 800], world_size=[20, 20], obstacles_corner=[],
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
def collision_response(c, v, m, size, obstacles, collision_coefficient=0.8, timestep=0.01):
    '''
    检测物体相撞后的位置和速度，物体均是轴对称的（长方形）
    n: 小车的数量
    m: 障碍物的数量
    c: 小车的四个角标, shape=(n,4,2)
    v: 小车的速度, shape=(n,2,)
    m: 小车的质量, shape=(n,)
    obstacles: 障碍物的四个角标, shape=(m,4,2)
    collision_coefficient: 碰撞系数, shape=1
    timestep: 时间步长, shape=1
    return: 小车相撞后的位置, shape=(n,4,2)
    return: 小车相撞后的速度, shape=(n,2,)
    '''
    # 先用包围盒算法粗略检测是否重叠
    # 先计算包围盒
    num_cars = c.shape[0]
    num_obstacles = obstacles.shape[0]

    # 向量化计算小车包围盒
    car_min_x = np.min(c[:, :, 0], axis=1)
    car_max_x = np.max(c[:, :, 0], axis=1)
    car_min_y = np.min(c[:, :, 1], axis=1)
    car_max_y = np.max(c[:, :, 1], axis=1)

    # 向量化计算障碍物包围盒
    obs_min_x = np.min(obstacles[:, :, 0], axis=1)
    obs_max_x = np.max(obstacles[:, :, 0], axis=1)
    obs_min_y = np.min(obstacles[:, :, 1], axis=1)
    obs_max_y = np.max(obstacles[:, :, 1], axis=1)

    #################################################
    # 小车与障碍物
    #################################################
    # 广播以检查所有小车和障碍物的组合
    car_min_x = car_min_x[:, np.newaxis]
    car_max_x = car_max_x[:, np.newaxis]
    car_min_y = car_min_y[:, np.newaxis]
    car_max_y = car_max_y[:, np.newaxis]

    # 计算是否重叠
    overlaps_x = (car_min_x < obs_max_x) & (car_max_x > obs_min_x)
    overlaps_y = (car_min_y < obs_max_y) & (car_max_y > obs_min_y)
    overlaps = overlaps_x & overlaps_y

    for i in range(num_cars):
        for j in range(num_obstacles):
            if overlaps[i, j]:
                car = c[i]
                obstacle = obstacles[j]
                # 再用分离轴定理检测是否重叠
                result = sat_collision(car, obstacle)
                if result is not None:
                    # 再分离重叠的矩形
                    # c[i] = separate_overlap(car, obstacle)
                    insert_points1, insert_points2 = result
                    (c[i], v[i]), _ = collision_separation(c[i], v[i], m[i], size[i], insert_points1, obstacles[j], np.zeros_like(v[i]), np.inf, calculate_rectangle_size(obstacles[j]), insert_points2, collision_coefficient, timestep)
    
    ##################################################################
    # 小车与小车
    ##################################################################
    # 重新计算小车包围盒用于小车之间的检测
    car_min_x = np.min(c[:, :, 0], axis=1)
    car_max_x = np.max(c[:, :, 0], axis=1)
    car_min_y = np.min(c[:, :, 1], axis=1)
    car_max_y = np.max(c[:, :, 1], axis=1)

    # 广播以检查所有小车和小车的组合
    car_min_x_1 = car_min_x[:, np.newaxis]
    car_max_x_1 = car_max_x[:, np.newaxis]
    car_min_y_1 = car_min_y[:, np.newaxis]
    car_max_y_1 = car_max_y[:, np.newaxis]

    overlaps_x_car = (car_min_x_1 < car_max_x) & (car_max_x_1 > car_min_x)
    overlaps_y_car = (car_min_y_1 < car_max_y) & (car_max_y_1 > car_min_y)
    overlaps_car = overlaps_x_car & overlaps_y_car

    for i in range(num_cars):
        for j in range(i + 1, num_cars):  # 避免重复检查
            if overlaps_car[i, j]:
                car1 = c[i]
                car2 = c[j]
                # 再用分离轴定理检测是否重叠
                result = sat_collision(car1, car2)
                if result is not None:
                    # 再分离重叠的矩形
                    # c[i], c[j] = separate_overlap_car(car1, car2)
                    insert_points1, insert_points2 = result
                    (c[i], v[i]), (c[j], v[j]) = collision_separation(c[i], v[i], m[i], size[i], insert_points1, c[j], v[j], m[j], size[j], insert_points2, collision_coefficient, timestep)
                    

    return c, v


def collision_separation(c1, v1, m1, size1, insert_points1, c2, v2, m2, size2, insert_points2, collision_coefficient=0.8, timestep=0.01):
    """
    碰撞分离，根据碰撞法线，计算接触点，从而计算两个物体碰撞后的位置和速度（需考虑旋转）

    参数:
    c1 (numpy.ndarray): 物体1的四个角标，形状为 (4, 2)。
    v1 (numpy.ndarray): 物体1的速度，形状为 (2,)。
    m1 (float): 物体1的质量。
    size1 (tuple): 物体1的大小，形状为 (2,)。
    insert_points1 (numpy.ndarray): 物体1的插入点，形状为 (4,)，类型是 bool。
    c2 (numpy.ndarray): 物体2的四个角标，形状为 (4, 2)。
    v2 (numpy.ndarray): 物体2的速度，形状为 (2,)。
    m2 (float): 物体2的质量。
    size2 (tuple): 物体2的大小，形状为 (2,)。
    insert_points2 (numpy.ndarray): 物体2的插入点，形状为 (4,)，类型是 bool。
    collision_coefficient (float, 可选): 碰撞系数，默认为 0.8。
    timestep (float, 可选): 时间步长，默认为 0.01。

    返回:
    tuple: 包含物体1和物体2碰撞后的位置和速度的元组。
    """
    # 旋转
    def calculate_rotation_angle(corner, center, m, size, insert_points, v, timestep, collision_coefficient):
        # 计算插入点中点
        insert_point = np.mean(insert_points, axis=0)
        
        # 计算相对位置向量（从质心到碰撞点）
        r_vector = insert_point - center
        
        # 计算转动惯量
        width = size[0]
        height = size[1]
        I = m * (width**2 + height**2) / 12  # 矩形转动惯量公式
        
        # 计算扭矩（r × F，这里F近似为速度方向）
        torque = np.cross(r_vector, v)
        
        # 计算角加速度（Δω = τ/I * timestep）
        rotation_factor = torque / (I + 1e-6) * timestep * collision_coefficient
        
        # 计算最终旋转角度（限制最大旋转幅度）
        return rotation_factor


    # 首先，我们需要计算两个多边形的质心
    centroid1 = np.mean(c1, axis=0)
    centroid2 = np.mean(c2, axis=0)

    # 计算旋转角度
    rotation_angle1 = calculate_rotation_angle(c1, centroid1, m1, size1, insert_points2, v2, timestep, collision_coefficient)
    rotation_angle2 = calculate_rotation_angle(c2, centroid2, m2, size2, insert_points1, v1, timestep, collision_coefficient)
    # 构建旋转矩阵
    rotation_matrix1 = np.array([
        [np.cos(rotation_angle1), -np.sin(rotation_angle1)],
        [np.sin(rotation_angle1), np.cos(rotation_angle1)] 
    ])
    rotation_matrix2 = np.array([
        [np.cos(rotation_angle2), -np.sin(rotation_angle2)],
        [np.sin(rotation_angle2), np.cos(rotation_angle2)] 
    ])
    
    # 将角标相对于质心的位置旋转
    car1_separated_relative = c1 - centroid1
    car1_separated_rotated_relative = np.dot(car1_separated_relative, rotation_matrix1)

    # 将旋转后的角标位置还原到原坐标系
    c1 = car1_separated_rotated_relative + centroid1

    if m2 != np.inf:
        car2_separated_relative = c2 - centroid2
        car2_separated_rotated_relative = np.dot(car2_separated_relative, rotation_matrix2)
        c2 = car2_separated_rotated_relative + centroid2

    # 对两个多边形的速度进行旋转
    v1 = np.dot(v1, rotation_matrix1)
    v2 = np.dot(v2, rotation_matrix2) if m2!= np.inf else v2

    # 速度减小
    v1 = v1 * collision_coefficient
    v2 = v2 * collision_coefficient


    return (c1, v1), (c2, v2)






def is_overlapping(poly1, poly2):
    """
    使用分离轴定理检测两个凸多边形是否重叠
    """
    polygons = [poly1, poly2]
    for polygon in polygons:
        # 计算所有边的法线
        edges = np.roll(polygon, -1, axis=0) - polygon
        normals = np.array([edges[:, 1], -edges[:, 0]]).T
        
        # 计算两个多边形在每个法线上的投影
        projections1 = np.dot(poly1, normals.T)
        projections2 = np.dot(poly2, normals.T)
        
        # 计算每个法线投影的最小值和最大值
        min1 = np.min(projections1, axis=0)
        max1 = np.max(projections1, axis=0)
        min2 = np.min(projections2, axis=0)
        max2 = np.max(projections2, axis=0)
        
        # 检查是否存在不重叠的投影
        if np.any(np.maximum(min1, min2) > np.minimum(max1, max2)):
            return False
    
    return True


def separate_overlap(car, other):
    """
    分离两个重叠的多边形
    """
    dx = np.mean(other[:, 0]) - np.mean(car[:, 0])
    dy = np.mean(other[:, 1]) - np.mean(car[:, 1])
    if dx != 0:
        car[:, 0] -= np.sign(dx) * 0.1
    if dy != 0:
        car[:, 1] -= np.sign(dy) * 0.1
    return car

def separate_overlap_and_rotate(car_corner, car_v, obstacle):
    """
    分离两个重叠的多边形
    第一个矩形向反方向移动的同时, 根据重叠的部分, 旋转第一个矩形

    参数:
    car_corner (numpy.ndarray): 小车矩形的四个角的坐标，形状为 (4, 2)。
    car_v (numpy.ndarray): 小车的速度向量，形状为 (2,)。
    obstacle (numpy.ndarray): 障碍物矩形的四个角的坐标，形状为 (4, 2)。

    返回:
    tuple: 分离并旋转后的小车矩形的四个角的坐标和速度向量。
    """
    def rotate_and_move(car_corner, car_v, diff, axis):
        if diff != 0:
            # 移动
            car_corner[:, axis] -= np.sign(diff) * 0.1
            # 动态调整旋转角度
            rotation_angle = np.sign(diff) * 0.01 * np.abs(diff)
            # 构建旋转矩阵
            rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                        [np.sin(rotation_angle), np.cos(rotation_angle)]])
            # 对小车矩形的四个角的坐标进行旋转
            car_corner = np.dot(car_corner, rotation_matrix)
            # 对小车的速度向量进行旋转
            car_v = np.dot(car_v, rotation_matrix)
        return car_corner, car_v

    # 计算小车和障碍物在 x 轴方向的中心位置差值
    dx = np.mean(obstacle[:, 0]) - np.mean(car_corner[:, 0])
    # 计算小车和障碍物在 y 轴方向的中心位置差值
    dy = np.mean(obstacle[:, 1]) - np.mean(car_corner[:, 1])

    # 处理 x 轴方向
    car_corner, car_v = rotate_and_move(car_corner, car_v, dx, 0)
    # 处理 y 轴方向
    car_corner, car_v = rotate_and_move(car_corner, car_v, dy, 1)

    return car_corner, car_v

def separate_overlap_and_rotate_car(car1_corner, car1_v, car2_corner, car2_v):
    """
    分离两个重叠的多边形
    第一个矩形向反方向移动的同时, 根据重叠的部分, 旋转第一个矩形
    第二个矩形向反方向移动的同时, 根据重叠的部分, 旋转第二个矩形

    参数:
    car1_corner (numpy.ndarray): 第一个小车矩形的四个角的坐标，形状为 (4, 2)。
    car1_v (numpy.ndarray): 第一个小车的速度向量，形状为 (2,)。
    car2_corner (numpy.ndarray): 第二个小车矩形的四个角的坐标，形状为 (4, 2)。
    car2_v (numpy.ndarray): 第二个小车的速度向量，形状为 (2,)。

    返回:
    tuple: 分离并旋转后的两个小车矩形的四个角的坐标和速度向量。
    """
    def rotate_and_move(car_corner, car_v, diff, axis):
        if diff != 0:
            # 移动
            car_corner[:, axis] -= np.sign(diff) * 0.1
            # 动态调整旋转角度
            rotation_angle = np.sign(diff) * 0.01 * np.abs(diff)
            # 构建旋转矩阵
            rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                        [np.sin(rotation_angle), np.cos(rotation_angle)]])
            # 对小车矩形的四个角的坐标进行旋转
            car_corner = np.dot(car_corner, rotation_matrix)
            # 对小车的速度向量进行旋转
            car_v = np.dot(car_v, rotation_matrix)
        return car_corner, car_v

    # 计算两个小车在 x 轴方向的中心位置差值
    dx = np.mean(car2_corner[:, 0]) - np.mean(car1_corner[:, 0])
    # 计算两个小车在 y 轴方向的中心位置差值
    dy = np.mean(car2_corner[:, 1]) - np.mean(car1_corner[:, 1])

    # 处理第一个小车在 x 轴方向
    car1_corner, car1_v = rotate_and_move(car1_corner, car1_v, dx, 0)
    # 处理第一个小车在 y 轴方向
    car1_corner, car1_v = rotate_and_move(car1_corner, car1_v, dy, 1)

    # 反向处理第二个小车在 x 轴方向
    car2_corner, car2_v = rotate_and_move(car2_corner, car2_v, -dx, 0)
    # 反向处理第二个小车在 y 轴方向
    car2_corner, car2_v = rotate_and_move(car2_corner, car2_v, -dy, 1)

    return car1_corner, car1_v, car2_corner, car2_v


def separate_overlap_car(car1, car2):
    """
    分离两个重叠的多边形
    两个多边形向反方向移动
    """
    # 首先，我们需要计算两个多边形的质心
    centroid1 = np.mean(car1, axis=0)
    centroid2 = np.mean(car2, axis=0)

    # 计算从质心1到质心2的向量
    separation_vector = centroid2 - centroid1

    # 归一化分离向量
    separation_vector = separation_vector / np.linalg.norm(separation_vector)

    # 假设我们每次分离移动一个固定的小距离，这里设为0.1
    separation_distance = 0.1

    # 两个多边形向反方向移动
    car1_separated = car1 - separation_vector * separation_distance
    car2_separated = car2 + separation_vector * separation_distance

    return car1_separated, car2_separated


def resolve_overlaps_vector(cars_corners, obstacles):
    """
    分离重叠的矩形
    cars_corners: 小车的四个角标, shape=(n, 4, 2)
    obstacles: 障碍物的四个角标, shape=(m, 4, 2)
    return: 分离重叠的矩形后的小车的四个角标, shape=(n, 4, 2)
    """
    # 先用包围盒算法粗略检测是否重叠
    # 先计算包围盒
    num_cars = cars_corners.shape[0]
    num_obstacles = obstacles.shape[0]

    # 向量化计算小车包围盒
    car_min_x = np.min(cars_corners[:, :, 0], axis=1)
    car_max_x = np.max(cars_corners[:, :, 0], axis=1)
    car_min_y = np.min(cars_corners[:, :, 1], axis=1)
    car_max_y = np.max(cars_corners[:, :, 1], axis=1)

    # 向量化计算障碍物包围盒
    obs_min_x = np.min(obstacles[:, :, 0], axis=1)
    obs_max_x = np.max(obstacles[:, :, 0], axis=1)
    obs_min_y = np.min(obstacles[:, :, 1], axis=1)
    obs_max_y = np.max(obstacles[:, :, 1], axis=1)

    #################################################
    # 小车与障碍物
    #################################################
    # 广播以检查所有小车和障碍物的组合
    car_min_x = car_min_x[:, np.newaxis]
    car_max_x = car_max_x[:, np.newaxis]
    car_min_y = car_min_y[:, np.newaxis]
    car_max_y = car_max_y[:, np.newaxis]

    # 计算是否重叠
    overlaps_x = (car_min_x < obs_max_x) & (car_max_x > obs_min_x)
    overlaps_y = (car_min_y < obs_max_y) & (car_max_y > obs_min_y)
    overlaps = overlaps_x & overlaps_y

    for i in range(num_cars):
        for j in range(num_obstacles):
            if overlaps[i, j]:
                car = cars_corners[i]
                obstacle = obstacles[j]
                # 再用分离轴定理检测是否重叠
                result = sat_collision(car, obstacle)
                if result is not None:
                    # 再分离重叠的矩形
                    cars_corners[i] = separate_overlap(car, obstacle)
    
    ##################################################################
    # 小车与小车
    ##################################################################
    # 重新计算小车包围盒用于小车之间的检测
    car_min_x = np.min(cars_corners[:, :, 0], axis=1)
    car_max_x = np.max(cars_corners[:, :, 0], axis=1)
    car_min_y = np.min(cars_corners[:, :, 1], axis=1)
    car_max_y = np.max(cars_corners[:, :, 1], axis=1)

    # 广播以检查所有小车和小车的组合
    car_min_x_1 = car_min_x[:, np.newaxis]
    car_max_x_1 = car_max_x[:, np.newaxis]
    car_min_y_1 = car_min_y[:, np.newaxis]
    car_max_y_1 = car_max_y[:, np.newaxis]

    overlaps_x_car = (car_min_x_1 < car_max_x) & (car_max_x_1 > car_min_x)
    overlaps_y_car = (car_min_y_1 < car_max_y) & (car_max_y_1 > car_min_y)
    overlaps_car = overlaps_x_car & overlaps_y_car

    for i in range(num_cars):
        for j in range(i + 1, num_cars):  # 避免重复检查
            if overlaps_car[i, j]:
                car1 = cars_corners[i]
                car2 = cars_corners[j]
                # 再用分离轴定理检测是否重叠
                result = sat_collision(car1, car2)
                if result is not None:
                    # 再分离重叠的矩形
                    cars_corners[i], cars_corners[j] = separate_overlap_car(car1, car2)

    return cars_corners







def resolve_overlaps(cars_corners, obstacles):
    n = len(cars_corners)
    m = len(obstacles)
    resolved_cars = cars_corners.copy()

    # 检测小车与障碍物的重叠
    for i in range(n):
        for j in range(m):
            nn = 0
            while is_overlapping(resolved_cars[i], obstacles[j]):
                resolved_cars[i] = separate_overlap(resolved_cars[i], obstacles[j])
                nn += 1
                if nn > 10:
                    break

    # 检测小车与小车的重叠
    for i in range(n):
        for j in range(i + 1, n):
            nn = 0
            while is_overlapping(resolved_cars[i], resolved_cars[j]):
                resolved_cars[i] = separate_overlap(resolved_cars[i], resolved_cars[j])
                resolved_cars[j] = separate_overlap(resolved_cars[j], resolved_cars[i])
                nn += 1
                if nn > 10:
                    break

    return resolved_cars

#############################################################################################
# 碰撞测试
#############################################################################################
def test_collision_response(q1, q2, v1, v2, m1, m2, collision_coefficient=0.8, friction_coefficient=0.8, timestep=0.01, totaltime=1,
                             test_model="GUI", videoLog=False):
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

    # 记录视频
    if videoLog:
        video_logger = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "collision_video.mp4")
        time.sleep(1)

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

    # 停止记录视频
    if videoLog:
        p.stopStateLogging(video_logger)
        # 保存视频
        p.saveStateLogging(video_logger, "collision_video.mp4")


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


###################################################################################
# 匀速圆周运动的计算
###################################################################################
def calculate_circular_motion(n, q, v, vNext, a, timestep):
    """
    计算多个小车在一段时间内的匀速圆周运动的最终速度、最终位置和经过的时间。

    参数:
    n (int): 小车的数量。
    q (numpy.ndarray): 小车的初始位置，形状为 (n, 2)。
    v (numpy.ndarray): 小车的初始速度，形状为 (n, 2)。
    vNext (numpy.ndarray): 小车的目标速度，形状为 (n, 2)。
    a (numpy.ndarray): 小车的向心加速度，形状为 (n,)。
    timestep (numpy.ndarray): 每个小车的时间步长，形状为 (n,)。

    返回:
    final_velocities (numpy.ndarray): 每个小车的最终速度，形状为 (n, 2)。
    final_positions (numpy.ndarray): 每个小车的最终位置，形状为 (n, 2)。
    elapsed_times (numpy.ndarray): 每个小车达到最终状态所经过的时间，形状为 (n,)。
    """
    # 初始化结果数组
    final_velocities = np.zeros((n, 2))
    final_positions = np.zeros((n, 2))
    elapsed_times = np.zeros(n)

    for i in range(n):
        # 获取当前小车的信息
        current_q = q[i]
        current_v = v[i]
        current_vNext = vNext[i]
        current_a = a[i]
        current_timestep = timestep[i]

        # 计算初速度和目标速度的夹角
        dot_product = np.dot(current_v, current_vNext)
        norm_v = np.linalg.norm(current_v)
        norm_vNext = np.linalg.norm(current_vNext)
        cos_angle = dot_product / (norm_v * norm_vNext)
        angle = np.arccos(cos_angle)

        # 计算角速度
        angular_velocity = current_a / norm_v

        # 计算达到目标速度所需的时间
        required_time = angle / angular_velocity

        # 确定实际使用的时间
        actual_time = min(required_time, current_timestep)

        # 计算旋转角度
        rotation_angle = angular_velocity * actual_time

        # 确定旋转方向
        cross_product = np.cross(current_v, current_vNext)
        if cross_product < 0:
            # 负的叉积表示逆时针旋转
            rotation_angle = -rotation_angle

        # 构建旋转矩阵
        rotation_matrix = np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle), np.cos(rotation_angle)]
        ])

        # 计算最终速度
        final_velocity = np.dot(rotation_matrix, current_v)

        # 计算圆周运动的半径
        radius = norm_v ** 2 / current_a

        # 计算圆心位置
        center_direction = np.array([-current_v[1], current_v[0]])
        center_direction = center_direction / np.linalg.norm(center_direction)
        # 根据旋转方向调整圆心方向
        if cross_product < 0:
            center_direction = -center_direction
        center = current_q + center_direction * radius

        # 计算最终位置
        initial_angle = np.arctan2(current_q[1] - center[1], current_q[0] - center[0])
        final_angle = initial_angle + rotation_angle
        final_position = center + radius * np.array([np.cos(final_angle), np.sin(final_angle)])

        # 保存结果
        final_velocities[i] = final_velocity
        final_positions[i] = final_position
        elapsed_times[i] = actual_time

    return final_velocities, final_positions, elapsed_times

def calculate_circular_motion_vector(q, v, vNext, a, timestep):
    """
    计算多个小车在一段时间内的匀速圆周运动的最终速度、最终位置和经过的时间。

    参数:
    n (int): 小车的数量。
    q (numpy.ndarray): 小车的初始位置，形状为 (n, 2)。
    v (numpy.ndarray): 小车的初始速度，形状为 (n, 2)。
    vNext (numpy.ndarray): 小车的目标速度，形状为 (n, 2)。
    a (numpy.ndarray): 小车的向心加速度，形状为 (n,)。
    timestep (numpy.ndarray): 每个小车的时间步长，形状为 (n,)。

    返回:
    final_velocities (numpy.ndarray): 每个小车的最终速度，形状为 (n, 2)。
    final_positions (numpy.ndarray): 每个小车的最终位置，形状为 (n, 2)。
    elapsed_times (numpy.ndarray): 每个小车达到最终状态所经过的时间，形状为 (n,)。
    """
    # from log import log
    # log(q)
    # log(v)
    # log(vNext)
    # log(a)
    # log(timestep)
    # 计算初速度和目标速度的夹角
    dot_product = np.sum(v * vNext, axis=1)
    norm_v = np.linalg.norm(v, axis=1)
    norm_vNext = np.linalg.norm(vNext, axis=1)
    cos_angle = dot_product / (norm_v * norm_vNext)
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.arccos(cos_angle)

    # 计算角速度
    angular_velocity = a / norm_v

    # 计算达到目标速度所需的时间
    required_time = angle / angular_velocity

    # 确定实际使用的时间
    actual_time = np.minimum(required_time, timestep)

    # 计算旋转角度
    rotation_angle = angular_velocity * actual_time

    # 确定旋转方向
    v_3d = np.pad(v, ((0, 0), (0, 1)), mode='constant')
    vNext_3d = np.pad(vNext, ((0, 0), (0, 1)), mode='constant')
    cross_product = np.cross(v_3d, vNext_3d)[:, 2]
    rotation_angle = np.where(cross_product < 0, -rotation_angle, rotation_angle)

    # 构建旋转矩阵
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)
    rotation_matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ]).transpose(2, 0, 1)

    # 计算最终速度
    final_velocities = np.einsum('nij,nj->ni', rotation_matrix, v)

    # 计算圆周运动的半径
    radius = norm_v ** 2 / a

    # 计算圆心方向
    center_direction = np.stack([-v[:, 1], v[:, 0]], axis=1)
    center_direction = center_direction / np.linalg.norm(center_direction, axis=1, keepdims=True)

    # 将 cross_product < 0 扩展为形状 (n, 2)
    condition = np.stack([cross_product < 0, cross_product < 0], axis=1)

    # 根据条件调整圆心方向
    center_direction = np.where(condition, -center_direction, center_direction)

    # 计算圆心位置
    center = q + center_direction * radius[:, np.newaxis]

    # 计算初始角度
    initial_angle = np.arctan2(q[:, 1] - center[:, 1], q[:, 0] - center[:, 0])

    # 计算最终角度
    final_angle = initial_angle + rotation_angle

    # 计算最终位置
    final_positions = center + radius[:, np.newaxis] * np.stack([np.cos(final_angle), np.sin(final_angle)], axis=1)

    # 保存经过的时间
    elapsed_times = actual_time

    return final_velocities, final_positions, elapsed_times

# 测试 calculate_circular_motion 方法
import timeit
def test_calculate_circular_motion():
    # 定义测试数据
    n = 10
    q = np.array([[-9.60790508, -3.78432431],
 [ 3.28626966,  5.33966847],
 [ 9.47037248,  0.6675621 ],
 [ 0.08648271,  4.38895002],
 [-8.36577298,  0.4531759 ],
 [-9.75458214,  3.70149102],
 [ 3.45225559, -1.34266393],
 [-9.06657077, -4.99149052],
 [ 9.5743255,  -5.31089035],
 [ 6.13016863, -4.71511016]])
    v = np.array(
        [[ 1.04959543e-11,  4.88859381e-11],
 [ 4.92199013e-11,  8.79780183e-12],
 [-3.53158460e-11, -3.53947880e-11],
 [ 5.68287463e-12, -4.96759996e-11],
 [ 4.79176887e-11, -1.42791845e-11],
 [ 1.61435633e-11, -4.73221446e-11],
 [-2.34590483e-11, -4.41551022e-11],
 [ 2.47303857e-11, -4.34558169e-11],
 [ 1.78494968e-11,  4.67054115e-11],
 [-4.67107771e-11,  1.78354508e-11]]
    )
    vNext = np.array(
       [[ 0.10495954,  0.48885938],
 [ 0.49219901,  0.08797802],
 [-0.35315846, -0.35394788],
 [ 0.05682875, -0.49676   ],
 [ 0.47917689, -0.14279185],
 [ 0.16143563, -0.47322145],
 [-0.23459048, -0.44155102],
 [ 0.24730386, -0.43455817],
 [ 0.17849497,  0.46705412],
 [-0.46710777,  0.17835451]] 
    )
    a = np.array(
        
    )
    timestep = np.array(
        
    )

    # 调用方法
    final_velocities, final_positions, elapsed_times = calculate_circular_motion(n, q, v, vNext, a, timestep)
    final_velocities = np.round(final_velocities, decimals=5)
    final_positions = np.round(final_positions, decimals=5)
    print("Final Velocities:", final_velocities)
    print("Final Positions:", final_positions)
    print("Elapsed Times:", elapsed_times)

    final_velocities_vector, final_positions_vector, elapsed_times_vector = calculate_circular_motion_vector(q, v, vNext, a, timestep)
    final_velocities_vector = np.round(final_velocities_vector, decimals=5)
    final_positions_vector = np.round(final_positions_vector, decimals=5)
    print("Final Velocities:", final_velocities_vector)
    print("Final Positions:", final_positions_vector)
    print("Elapsed Times:", elapsed_times_vector)

    # # 随机生成数据，测试两个函数生成的结果是否相同，各自耗费的时间是多少
    # num_tests = 100  # 测试次数
    # max_n = 100  # 最大小车数量
    # max_coord = 10  # 最大坐标值
    # max_acceleration = 10  # 最大加速度
    # max_timestep = 10  # 最大时间步长

    # for _ in range(num_tests):
    #     n = np.random.randint(1, max_n)
    #     q = np.random.uniform(-max_coord, max_coord, (n, 2))
    #     v = np.random.uniform(-max_coord, max_coord, (n, 2))
    #     vNext = np.random.uniform(-max_coord, max_coord, (n, 2))
    #     a = np.random.uniform(0, max_acceleration, n)
    #     timestep = np.random.uniform(0, max_timestep, n)

    #     # 测量 calculate_circular_motion 的执行时间
    #     start_time_loop = timeit.default_timer()
    #     final_velocities_loop, final_positions_loop, elapsed_times_loop = calculate_circular_motion(n, q, v, vNext, a, timestep)
    #     end_time_loop = timeit.default_timer()
    #     time_loop = end_time_loop - start_time_loop

    #     # 测量 calculate_circular_motion_vector 的执行时间
    #     start_time_vector = timeit.default_timer()
    #     final_velocities_vector, final_positions_vector, elapsed_times_vector = calculate_circular_motion_vector(q, v, vNext, a, timestep)
    #     end_time_vector = timeit.default_timer()
    #     time_vector = end_time_vector - start_time_vector

    #     # 比较结果
    #     assert np.allclose(final_velocities_loop, final_velocities_vector), "Final velocities do not match!"
    #     assert np.allclose(final_positions_loop, final_positions_vector), "Final positions do not match!"
    #     assert np.allclose(elapsed_times_loop, elapsed_times_vector), "Elapsed times do not match!"

    #     print(f"Test {_ + 1}:")
    #     print(f"  Loop time: {time_loop:.6f} seconds")
    #     print(f"  Vectorized time: {time_vector:.6f} seconds")
    #     print(f"  Speedup: {time_loop / time_vector:.2f}x")

# # 运行测试
# if __name__ == "__main__":
#     test_calculate_circular_motion()


###################################################################################
# 加速直线运动的计算（固定功率和力的最大值）
###################################################################################
def calculate_acceleration_motion(n, mass, q, v, vNext, power, force_max, friction_force, timestep):
    """
    计算多个小车在一段时间内的加速直线运动的最终速度、最终位置和经过的时间。

    参数:
    n (int): 小车的数量。
    mass (numpy.ndarray): 小车的质量，形状为 (n,)。
    q (numpy.ndarray): 小车的初始位置，形状为 (n, 2)。
    v (numpy.ndarray): 小车的初始速度，形状为 (n, 2)。
    vNext (numpy.ndarray): 小车的目标速度，形状为 (n, 2)。
    power (numpy.ndarray): 小车的功率，形状为 (n,)。
    force_max (numpy.ndarray): 小车的最大力，形状为 (n,)。
    friction_force (numpy.ndarray): 小车的摩擦力，形状为 (n,)。
    timestep (numpy.ndarray): 每个小车的时间步长，形状为 (n,)。

    返回:
    final_velocities (numpy.ndarray): 每个小车的最终速度，形状为 (n, 2)。
    final_positions (numpy.ndarray): 每个小车的最终位置，形状为 (n, 2)。
    elapsed_times (numpy.ndarray): 每个小车达到最终状态所经过的时间，形状为 (n,)。
    """
    # 初始化结果数组
    final_velocities = np.zeros((n, 2))
    final_positions = np.zeros((n, 2))
    elapsed_times = np.zeros(n)

    for i in range(n):
        # 获取当前小车的信息
        current_mass = mass[i]
        current_q = q[i]
        current_v = v[i]
        current_v_size = np.linalg.norm(current_v)
        current_vNext = vNext[i]
        current_vNext_size = np.linalg.norm(current_vNext)
        direction = current_vNext / np.linalg.norm(current_vNext)
        current_power = power[i]
        current_force_max = force_max[i]
        current_friction_force = friction_force[i]
        current_timestep = timestep[i]

        time_use = 0

        # 第一阶段：速度未达到，功率提供的动力大于force_max
        # 先计算临界速度
        vDiv = current_power / current_force_max
        vCondition = current_v_size < vDiv and time_use < current_timestep and current_v_size < current_vNext_size
        if vCondition:
            # 计算加速度
            a = (current_force_max - current_friction_force) / current_mass
            # 计算达到目标速度所需的时间
            vTarget = np.minimum(vDiv, current_vNext_size)
            required_time = (vTarget - current_v_size) / a
            # 确定实际使用的时间
            time_use += min(required_time, current_timestep)
            # 计算最终速度
            current_v_size = current_v_size + a * time_use
            current_v = current_v_size * direction

            # 计算最终位置
            current_q = current_q + current_v * time_use + 0.5 * a * direction * time_use ** 2
        # 第二阶段：速度达到临界速度，功率提供的动力小于force_max
        vCondition = current_v_size >= vDiv and time_use < current_timestep and current_v_size < current_vNext_size

        if vCondition:
            # 计算规定时间内能不能达到目标速度
            # 第一步计算目标速度是不是可以达到的
            v_max = current_power / current_friction_force - 1e-10
            vTarget = current_vNext_size if current_vNext_size < v_max else v_max

            # 第二步从v0计算t0
            # 提前计算所需参数
            c1 = -current_mass/current_friction_force
            c2 = current_mass*current_power/(current_friction_force**2)
            c3 = current_friction_force/current_power
            t0 = c1*current_v_size-c2*np.log(1-c3*current_v_size)
            # 第三步计算到达目标所需时间
            t1 = c1*vTarget-c2*np.log(1-c3*vTarget)
            # 第四步计算最终速度和所用时间
            # 如果可以达到，则正常达到
            single_time_use = 0
            if t1 - t0 < current_timestep - time_use:
                single_time_use = t1 - t0
                vActual = vTarget
            else:
                # 如果不可以达到，则用lambertw函数计算
                single_time_use = current_timestep - time_use
                t2 = t0 + single_time_use
                vActual = calculate_v(t2, current_mass, current_friction_force, current_power)
            
            time_use += single_time_use
            # 第五步计算最终位置
            x1 = current_power * single_time_use / current_friction_force
            x2 = current_mass * (vActual - current_v_size) * (vActual + current_v_size) / (2 * current_friction_force)
            front_size = x1 - x2

            current_q = current_q + direction * front_size
            current_v = vActual * direction
        
        final_velocities[i] = current_v
        final_positions[i] = current_q
        elapsed_times[i] = time_use

    return final_velocities, final_positions, elapsed_times

def calculate_acceleration_motion_vector(mass, q, v, vNext, power, force_max, friction_force, timestep):
    """
    计算多个小车在一段时间内的加速直线运动的最终速度、最终位置和经过的时间。

    参数:
    n (int): 小车的数量。
    mass (numpy.ndarray): 小车的质量，形状为 (n,)。
    q (numpy.ndarray): 小车的初始位置，形状为 (n, 2)。
    v (numpy.ndarray): 小车的初始速度，形状为 (n, 2)。
    vNext (numpy.ndarray): 小车的目标速度，形状为 (n, 2)。
    power (numpy.ndarray): 小车的功率，形状为 (n,)。
    force_max (numpy.ndarray): 小车的最大力，形状为 (n,)。
    friction_force (numpy.ndarray): 小车的摩擦力，形状为 (n,)。
    timestep (numpy.ndarray): 每个小车的时间步长，形状为 (n,)。

    返回:
    final_velocities (numpy.ndarray): 每个小车的最终速度，形状为 (n, 2)。
    final_positions (numpy.ndarray): 每个小车的最终位置，形状为 (n, 2)。
    elapsed_times (numpy.ndarray): 每个小车达到最终状态所经过的时间，形状为 (n,)。
    """
    n = len(mass)
    # 初始化结果数组
    final_velocities = np.zeros((n, 2))
    final_positions = np.zeros((n, 2))
    elapsed_times = np.zeros(n)

    # 计算速度大小和方向
    v_size = np.linalg.norm(v, axis=1)
    vNext_size = np.linalg.norm(vNext, axis=1)
    direction = vNext / vNext_size[:, np.newaxis]

    # 计算临界速度
    vDiv = power / force_max

    # 第一阶段：速度未达到，功率提供的动力大于force_max
    vCondition1 = (v_size < vDiv) & (elapsed_times < timestep) & (v_size < vNext_size)
    a = (force_max - friction_force) / mass
    vTarget1 = np.minimum(vDiv, vNext_size)
    required_time1 = (vTarget1 - v_size) / a
    time_use1 = np.minimum(required_time1, timestep)
    v_size[vCondition1] = v_size[vCondition1] + a[vCondition1] * time_use1[vCondition1]
    v[vCondition1] = v_size[vCondition1][:, np.newaxis] * direction[vCondition1]
    q[vCondition1] = q[vCondition1] + v[vCondition1] * time_use1[vCondition1][:, np.newaxis] + 0.5 * a[vCondition1][:, np.newaxis] * direction[vCondition1] * time_use1[vCondition1][:, np.newaxis]**2
    elapsed_times[vCondition1] += time_use1[vCondition1]

    # 第二阶段：速度达到临界速度，功率提供的动力小于force_max
    vCondition2 = (v_size >= vDiv) & (elapsed_times < timestep) & (v_size < vNext_size)
    v_max = power / friction_force - 1e-10
    vTarget2 = np.where(vNext_size < v_max, vNext_size, v_max)
    c1 = -mass / friction_force
    c2 = mass * power / (friction_force**2)
    c3 = friction_force / power
    t0 = c1 * v_size - c2 * np.log(1 - c3 * v_size)
    t1 = c1 * vTarget2 - c2 * np.log(1 - c3 * vTarget2)
    single_time_use = np.where(t1 - t0 < timestep - elapsed_times, t1 - t0, timestep - elapsed_times)
    t2 = t0 + single_time_use
    vActual = calculate_v(t2, mass, friction_force, power)
    x1 = power * single_time_use / friction_force
    x2 = mass * (vActual - v_size) * (vActual + v_size) / (2 * friction_force)
    front_size = x1 - x2
    q[vCondition2] = q[vCondition2] + direction[vCondition2] * front_size[vCondition2][:, np.newaxis]
    v[vCondition2] = vActual[vCondition2][:, np.newaxis] * direction[vCondition2]
    elapsed_times[vCondition2] += single_time_use[vCondition2]

    final_velocities = v
    final_positions = q

    return final_velocities, final_positions, elapsed_times


from scipy.special import lambertw
def calculate_v(t, m, f, P):
    """
    计算速度 v 作为时间 t 的函数
    参数:
        t : 时间（标量或数组）
        m, f, P : 物理常数（标量）
    返回:
        v : 速度值（实数部分）
    """
    exponent = - (f**2) / (m * P) * t
    arg = -np.exp(exponent - 1)  # -e^{exponent - 1} = - (e^{-1} * e^{exponent})
    w = lambertw(arg, k=0)       # 主分支(k=0)
    return (P / f) * (1 + w.real)  # 提取实数部分

# 测试函数
def test_calculate_acceleration_motion():
    n = 2
    mass = np.array([1.0, 2.0])
    q = np.array([[0.0, 0.0], [1.0, 1.0]])
    v = np.array([[1.0, 0.0], [0.0, 1.0]])
    vNext = np.array([[100.0, 0.0], [0.0, 200.0]])
    power = np.array([10.0, 20.0])
    force_max = np.array([5.0, 10.0])
    friction_force = np.array([1.0, 2.0])
    timestep = np.array([1.0, 2.0])

    # 调用函数
    final_velocities, final_positions, elapsed_times = calculate_acceleration_motion(n, mass, q, v, vNext, power, force_max, friction_force, timestep)
    # 打印结果
    print("Final Velocities:", final_velocities)
    print("Final Positions:", final_positions)
    print("Elapsed Times:", elapsed_times)

    final_velocities_vector, final_positions_vector, elapsed_times_vector = calculate_acceleration_motion_vector(n, mass, q, v, vNext, power, force_max, friction_force, timestep)
    # 打印结果
    print("Final Velocities:", final_velocities_vector)
    print("Final Positions:", final_positions_vector)
    print("Elapsed Times:", elapsed_times_vector)


##########################################################################################
# 匀减速直线运动的计算
##########################################################################################
def calculate_deceleration_motion(mass, q, v, vNext, force_max, friction_force, timestep):
    """
    计算多个小车在一段时间内的匀减速直线运动的最终速度、最终位置和经过的时间。

    参数:
    n (int): 小车的数量。
    mass (numpy.ndarray): 小车的质量，形状为 (n,)。
    q (numpy.ndarray): 小车的初始位置，形状为 (n, 2)。
    v (numpy.ndarray): 小车的初始速度，形状为 (n, 2)。
    vNext (numpy.ndarray): 小车的目标速度，形状为 (n, 2)。
    power (numpy.ndarray): 小车的功率，形状为 (n,)。
    force_max (numpy.ndarray): 小车的最大力，形状为 (n,)。
    friction_force (numpy.ndarray): 小车的摩擦力，形状为 (n,)。
    timestep (numpy.ndarray): 每个小车的时间步长，形状为 (n,)。

    返回:
    final_velocities (numpy.ndarray): 每个小车的最终速度，形状为 (n, 2)。
    final_positions (numpy.ndarray): 每个小车的最终位置，形状为 (n, 2)。
    elapsed_times (numpy.ndarray): 每个小车达到最终状态所经过的时间，形状为 (n,)。
    """
    n = len(mass)
    # 初始化结果数组
    final_velocities = np.zeros((n, 2))
    final_positions = np.zeros((n, 2))
    elapsed_times = np.zeros(n)

    # 计算力的大小
    force = force_max + friction_force

    # 计算速度大小和方向
    v_size = np.linalg.norm(v, axis=1)
    vNext_size = np.linalg.norm(vNext, axis=1)
    direction = -v / v_size[:, np.newaxis]

    # 计算加速度
    a = force / mass

    # 计算时间
    time_use = np.minimum(timestep, (v_size - vNext_size) / a)

    # 计算最终速度和位置
    # 使用NumPy向量化操作替代循环
    final_velocities = v + a[:, np.newaxis] * time_use[:, np.newaxis] * direction
    final_positions = q + v * time_use[:, np.newaxis] + 0.5 * a[:, np.newaxis] * direction * time_use[:, np.newaxis]**2
    elapsed_times = time_use


    return final_velocities, final_positions, elapsed_times

# 测试 calculate_deceleration_motion 方法
def test_calculate_deceleration_motion():
    # 定义测试数据
    n = 2
    mass = np.array([1.0, 2.0])
    q = np.array([[0.0, 0.0], [1.0, 1.0]])
    v = np.array([[1.0, 0.0], [0.0, 1.0]])
    vNext = np.array([[0.0, 0.0], [0.0, 0.0]])
    force_max = np.array([1.0, 2.0])
    friction_force = np.array([1, 2])
    timestep = np.array([0.5, 1.0])

    # 调用方法
    final_velocities, final_positions, elapsed_times = calculate_deceleration_motion(n, mass, q, v, vNext, force_max, friction_force, timestep)

    # 打印结果
    print("Final Velocities:", final_velocities)
    print("Final Positions:", final_positions)
    print("Elapsed Times:", elapsed_times)

# # 运行测试
# test_calculate_deceleration_motion()


#########################################################################################################
# 碰撞检测
#########################################################################################################

def sat_collision(c1, c2):
    """
    使用分离轴定理（SAT）检测两个矩形是否发生碰撞。

    参数:
    c1 (numpy.ndarray): 第一个矩形的四个角的坐标，形状为 (4, 2)。
    c2 (numpy.ndarray): 第二个矩形的四个角的坐标，形状为 (4, 2)。

    返回:
    tuple: 如果发生碰撞，返回在对方内部的点；如果未发生碰撞，返回 None。
    """
    # 存储所有可能的分离轴
    axes = []
    # 遍历第一个矩形的每条边
    for i in range(4):
        # 获取当前边的向量
        edge = c1[(i + 1) % 4] - c1[i]
        # 计算垂直于边的轴
        axis = np.array([-edge[1], edge[0]])
        # 归一化轴向量
        axis = axis / np.linalg.norm(axis)
        axes.append(axis)

    # 遍历第二个矩形的每条边
    for i in range(4):
        # 获取当前边的向量
        edge = c2[(i + 1) % 4] - c2[i]
        # 计算垂直于边的轴
        axis = np.array([-edge[1], edge[0]])
        # 归一化轴向量
        axis = axis / np.linalg.norm(axis)
        axes.append(axis)

    # 初始化在对方内部的点（全在）
    insert_points1 = np.ones(c1.shape[0], dtype=bool)
    insert_points2 = np.ones(c2.shape[0], dtype=bool)

    # 遍历所有分离轴
    for axis in axes:
        # 投影第一个矩形的顶点到当前轴上
        projections_c1 = np.dot(c1, axis)
        # 找到投影的最小值和最大值
        min_c1 = np.min(projections_c1)
        max_c1 = np.max(projections_c1)

        # 投影第二个矩形的顶点到当前轴上
        projections_c2 = np.dot(c2, axis)
        # 找到投影的最小值和最大值
        min_c2 = np.min(projections_c2)
        max_c2 = np.max(projections_c2)

        # 检查投影是否重叠
        if max_c1 < min_c2 or max_c2 < min_c1:
            # 如果没有重叠，则两个矩形没有碰撞
            return None

        # 计算那些点在对方内部
        insert_points1 = np.logical_and(insert_points1, (projections_c1 >= min_c2) & (projections_c1 <= max_c2))
        insert_points2 = np.logical_and(insert_points2, (projections_c2 >= min_c1) & (projections_c2 <= max_c1))

    return insert_points1, insert_points2
