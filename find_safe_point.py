import numpy as np
from sklearn.cluster import DBSCAN


class FindSafeSpot:
    def __init__(self, history_data, escape_current_position, obstacles_pos, delta_time, obstacles_size):
        """
        Initialize the trajectory predictor with initial conditions.

        Parameters:
            history_positions (dict of list of numpy.array): Historical positions of multiple vehicles.
            delta_time (float): Time interval between each step.
        """

        self.delta_time = delta_time
        self.positions = {vehicle_id: positions for vehicle_id, positions in history_data}
        self.velocities = {vehicle_id: self.calculate_velocities(vehicle_id) for vehicle_id in self.positions}

        self.escape_current_position = escape_current_position
        self.obstacles = []

        for i in range(len(obstacles_pos)):
            self.obstacles.append(np.array([obstacles_pos[i][0], obstacles_pos[i][1]]))

        self.obstacles_size = obstacles_size

    def calculate_velocities(self, vehicle_id):
        """
        Calculate velocities based on position differences.
        """
        positions = self.positions[vehicle_id]
        velocities = []
        for i in range(1, len(positions)):
            velocity = (positions[i] - positions[i - 1]) / self.delta_time
            velocities.append(velocity)
        velocities.append(velocities[-1] if velocities else np.array([0, 0]))  # Append last known velocity
        return velocities

    def is_collision(self, position):
        # 处理obstacles_size是单一值的情况
        if isinstance(self.obstacles_size, (list, tuple)):
            sizes = self.obstacles_size
        else:
            sizes = [self.obstacles_size] * len(self.obstacles)  # 创建与障碍物数量相等的尺寸列表

        for obstacle_center, size in zip(self.obstacles, sizes):
            if np.linalg.norm(position - obstacle_center) < size:
                return True
        return False

    def turn_direction(self, position, velocity, obstacle_center):
        # Calculate vector from position to obstacle
        to_obstacle = obstacle_center - position
        # Calculate vector perpendicular to velocity
        perpendicular = np.array([-velocity[1], velocity[0]])
        # Determine direction to turn based on dot product
        if np.dot(perpendicular, to_obstacle) > 0:
            # Turn right
            return np.array([velocity[1], -velocity[0]])
        else:
            # Turn left
            return np.array([-velocity[1], velocity[0]])

    def avoid_collision(self, position, velocity):
        if not isinstance(self.obstacles_size, (list, tuple)):
            sizes = [self.obstacles_size] * len(self.obstacles)
        else:
            sizes = self.obstacles_size

        for obstacle_center, size in zip(self.obstacles, sizes):
            if np.linalg.norm(position - obstacle_center) < size:
                # Calculate new direction
                return self.turn_direction(position, velocity, obstacle_center)
        return velocity

    def predict_trajectory(self, num_steps=6):
        predictions = {}
        for vehicle_id in self.positions:
            current_acceleration = self.calculate_acceleration(vehicle_id)
            current_position = np.array(self.positions[vehicle_id][-1], dtype=float)
            current_velocity = np.array(self.velocities[vehicle_id][-1], dtype=float)

            trajectory = [current_position.copy()]
            for _ in range(num_steps):
                current_velocity += current_acceleration * self.delta_time
                new_position = current_position + current_velocity * self.delta_time + 0.5 * current_acceleration * (
                            self.delta_time ** 2)
                # 检查并修正位置约束
                new_position = self.apply_boundaries(new_position)

                if self.is_collision(new_position):
                    current_velocity = self.avoid_collision(current_position, current_velocity)
                    new_position = current_position + current_velocity * self.delta_time + 0.5 * current_acceleration * (
                            self.delta_time ** 2)
                    # 再次应用边界约束
                    new_position = self.apply_boundaries(new_position)

                trajectory.append(new_position.copy())
                current_position = new_position
            predictions[vehicle_id] = np.array(trajectory)
        return predictions

    def apply_boundaries(self, position):
        """
        Apply boundaries to keep the position within the specified limits.
        """
        x_min, x_max = -11.5, 11.5
        y_min, y_max = -6.5, 6.5
        position[0] = max(x_min, min(x_max, position[0]))
        position[1] = max(y_min, min(y_max, position[1]))
        return position


    def calculate_acceleration(self, vehicle_id):
        """
        Calculate acceleration based on velocity differences.
        """
        velocities = self.velocities[vehicle_id]
        if len(velocities) < 2:
            return np.array([0, 0])  # Default acceleration for insufficient data
        velocity_previous = velocities[-2]
        velocity_current = velocities[-1]
        return (velocity_current - velocity_previous) / self.delta_time

    def find_safe_spot(self, predicted_trajectories, num_safe_points = 10, eps=15, min_samples=3):
        """
        :param predicted_trajectories: 预测的轨迹
        :param obstacles: 障碍物位置
        :param eps: 选择远离聚类中心的阈值距离
        :param min_samples: 聚类密度
        :return: 候选安全点
        """
        points = np.vstack(predicted_trajectories)
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = clustering.labels_

        # 修正错误的部分: 处理孤立轨迹点
        noise_indices = labels == -1
        noise_points = points[noise_indices]  # 孤立的噪声点
        directions = []
        for traj in predicted_trajectories:
            # 检查当前轨迹的所有点是否都是噪声点
            traj_points = np.vstack(traj)
            if np.all([np.any(np.all(noise_points == point, axis=1)) for point in traj_points]):
                if len(traj) > 1:
                    direction = traj[-1] - traj[0]
                    directions.append(direction)

        # 寻找安全点
        safe_points = []
        for _ in range(num_safe_points):  # 尝试生成多个安全点
            candidate_x = np.random.uniform(-11.5, 11.5)
            candidate_y = np.random.uniform(-6.5, 6.5)
            candidate = np.array([candidate_x, candidate_y])

            # candidate = np.random.rand(2) * 100
            if all(np.linalg.norm(candidate - ob) > self.obstacles_size for ob in self.obstacles):
                if not np.any([np.linalg.norm(candidate - center) < eps for center in directions]):
                    safe_points.append(candidate)

        if safe_points:  # 确保有安全点被找到
            distances = [np.linalg.norm(self.escape_current_position - point) for point in safe_points]
            min_index = np.argmin(distances)
            nearest_point = safe_points[min_index]
            return nearest_point


