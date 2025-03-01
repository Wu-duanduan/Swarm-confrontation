import cv2
import numpy as np

# 基于距离阈值 实现车辆轨迹跟踪


class CarTracker:
    def __init__(self):
        self.tracks_red = []  # 存储红车的轨迹
        self.tracks_blue = []  # 存储蓝车的轨迹
        self.next_car_id = 0  # 车辆的唯一ID

    def track(self, red_positions, blue_positions, frame_count):
        """
        根据当前位置更新车辆轨迹
        """
        # 更新红车轨迹
        # id 顺序 先红后蓝 x轴递增 x小在前
        self._update_tracks(self.tracks_red, red_positions, "red", frame_count)

        # 更新蓝车轨迹
        self._update_tracks(self.tracks_blue, blue_positions, "blue", frame_count)
        # print(self.tracks_red, self.tracks_blue)
        return self.tracks_red, self.tracks_blue

    def draw_tracks(self, image, tracks, color=(0, 255, 0), save_flag=False, save_path=None):
        """绘制轨迹"""
        image = np.copy(image)  # 原图只读
        for track in tracks:
            positions = track['positions']
            for i in range(1, len(positions)):
                # 获取相邻的两个点
                prev_pos = positions[i - 1]
                curr_pos = positions[i]

                # 绘制轨迹线
                cv2.line(image, prev_pos, curr_pos, color, 2)
            if save_flag:
                cv2.imwrite(save_path, image)
                #print("Image saved to", save_path)

        return image

    def calculate_speed_per_frame(self, track):
        """
        计算目标的像素每帧速度
        :param track: 轨迹数据，包含历史位置
        :return: 速度 (vx, vy)
        """
        if len(track['positions']) < 2:
            return 0, 0  # 如果轨迹点少于2个，无法计算速度

        # 获取最近两个位置
        prev_pos = track['positions'][-2]
        curr_pos = track['positions'][-1]

        # 计算位置变化量
        delta_x = curr_pos[0] - prev_pos[0]
        delta_y = curr_pos[1] - prev_pos[1]

        # 速度就是每帧的位置变化量
        vx = delta_x  # 像素每帧
        vy = delta_y  # 像素每帧

        return vx, vy

    def _update_tracks(self, tracks, current_positions, color, frame_count):
        if not current_positions:
            return

        new_tracks = []
        matched = [False] * len(current_positions)

        # 尝试匹配现有轨迹
        for track in tracks:
            #closest_dist = float('inf')
            closest_dist = 50
            closest_idx = -1
            for i, pos in enumerate(current_positions):
                dx = pos[0] - track['last_position'][0]
                dy = pos[1] - track['last_position'][1]
                dist = np.sqrt(dx ** 2 + dy ** 2)

                if dist < closest_dist and not matched[i]:
                    closest_dist = dist
                    closest_idx = i

            if closest_idx != -1:
                track['positions'].append(current_positions[closest_idx])
                track['last_position'] = current_positions[closest_idx]
                track['last_seen_frame'] = frame_count
                matched[closest_idx] = True
                new_tracks.append(track)

        # 添加未匹配的新轨迹
        for i, pos in enumerate(current_positions):
            if not matched[i]:
                new_tracks.append({
                    'id': self.next_car_id,
                    'color': color,
                    'positions': [pos],
                    'last_position': pos,
                    'first_seen_frame': frame_count,
                    'last_seen_frame': frame_count
                })
                self.next_car_id += 1

        # 移除消失的轨迹
        current_frame = frame_count
        new_tracks = [
            track for track in new_tracks
            if track['last_seen_frame'] + 3 >= current_frame  # 失踪3帧视为消失
        ]

        tracks[:] = new_tracks
