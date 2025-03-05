"""
2D rendering framework
"""
from __future__ import division
import os
import six
import sys

if "Apple" in sys.version:
    if 'DYLD_FALLBACK_LIBRARY_PATH' in os.environ:
        os.environ['DYLD_FALLBACK_LIBRARY_PATH'] += ':/usr/lib'
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite

# from gym.utils import reraise
from gym import error

try:
    import pyglet
except ImportError as e:
    print(
        suffix="HINT: you can install pyglet directly via 'pip install pyglet'. But if you really just want to install all Gym dependencies and not have to think about it, 'pip install -e .[all]' or 'pip install gym[all]' will do it.")

try:
    from pyglet.gl import *
except ImportError as e:
    print(prefix="Error occured while running `from pyglet.gl import *`",
          suffix="HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'. If you're running on a server, you may need a virtual frame buffer; something like this should work: 'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'")

import math
import numpy as np

RAD2DEG = 57.29577951308232

def apply_transform(pos, transform):
    x, y = pos
    newx = x * transform.scale[0] * np.cos(transform.rotation) - y * transform.scale[1] * np.sin(transform.rotation) + transform.translation[0]
    newy = x * transform.scale[0] * np.sin(transform.rotation) + y * transform.scale[1] * np.cos(transform.rotation) + transform.translation[1]
    return (newx, newy)

def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.

    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error('Invalid display specification: {}. (Must be a string like :0 or None.)'.format(spec))


class Viewer(object):
    def __init__(self, width, height, display=None):
        display = get_display(display)

        self.width = width
        self.height = height

        self.window = pyglet.window.Window(width=width, height=height, display=display)
        self.window.on_close = self.window_closed_by_user
        self.geoms = []
        self.onetime_geoms = []
        self.transform = Transform()
        self.light_source = (0, 0)

        glEnable(GL_BLEND)
        # glEnable(GL_MULTISAMPLE)
        glEnable(GL_LINE_SMOOTH)
        # glHint(GL_LINE_SMOOTH_HINT, GL_DONT_CARE)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glLineWidth(3.0)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.close()

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        self.transform = Transform(
            translation=(-left * scalex, -bottom * scaley),
            scale=(scalex, scaley))

    def camera_follow(self, ego_pos, ego_yaw, FOV, detect_range):
        # ego_pos * R * scale + T = (w/2, 0)
        # 因为R先于Scale，所以当缩放比例不同时，坐标变换后会有形变
        # 例如，当scale_x=1, scale_y=2时，正方形旋转拉伸会变成菱形
        # 为了避免形变，需要保证缩放比例相同
        # [x', y'].T = [x, y].T * R * S * R.inv + T 需要反解出角度和拉伸比例
        # 调整小车朝向为y轴方向（假设原始yaw为x轴方向）
        ego_yaw = ego_yaw - np.pi / 2
        
        # 计算相机的水平和垂直视野范围
        x_range = detect_range * np.tan(FOV / 2) * 2
        y_range = detect_range
        
        # 计算缩放比例，将世界坐标映射到屏幕像素
        scale_x = self.width / x_range
        scale_y = self.height / y_range
        assert scale_x == scale_y  
        # 计算旋转角度
        rotation = -ego_yaw

        # 计算平移量，将小车位置调整到屏幕底部中心              
        ego_pos_prime = np.array([ego_pos[0] * np.cos(rotation) - ego_pos[1] * np.sin(rotation),
                                    ego_pos[0] * np.sin(rotation) + ego_pos[1] * np.cos(rotation)])
        transform = np.array([self.width / 2, 0]) / np.array([scale_x, scale_y]) - ego_pos_prime

        # 设置变换参数
        self.transform.set_translation(transform[0] * scale_x, transform[1] * scale_y)
        self.transform.set_rotation(rotation)
        self.transform.set_scale(scale_x, scale_y)

    def draw_shadow(self, idx=0):
        light_x, light_y = self.light_source

        for geom in self.geoms[idx:]:
            if isinstance(geom, FilledPolygon):  # 只对障碍物计算阴影
                vertices = geom.v.copy()  # 获取多边形顶点
                shadow_edges = []

                for i in range(len(vertices)):
                    vertices[i] = apply_transform(vertices[i], geom.attrs[1])


                for v in vertices:
                    angle = np.arctan2(v[1] - light_y, v[0] - light_x)
                    shadow_edges.append((v, angle))

                # 按角度排序，确保阴影多边形的顶点顺序正确
                shadow_edges.sort(key=lambda x: x[1])
                # 计算阴影投影点（将点沿着光源方向延展）
                shadow_points = []
                if len(shadow_edges) == 4:
                    shadow_points.append(shadow_edges[0][0])  # 添加第一个点

                    dist_from_light1 = np.sqrt((shadow_edges[0][0][0] - light_x) ** 2 + (shadow_edges[0][0][1] - light_y) ** 2)
                    dist_from_light2 = np.sqrt((shadow_edges[-1][0][0] - light_x) ** 2 + (shadow_edges[-1][0][1] - light_y) ** 2)
                    dist_far = max(dist_from_light1, dist_from_light2)
                    # 计算中间的两个点
                    for v, angle in shadow_edges[1:3]:
                        dist = np.sqrt((v[0] - light_x) ** 2 + (v[1] - light_y) ** 2)
                        if dist > dist_far:
                            shadow_points.append(v)

                    shadow_points.append(shadow_edges[-1][0])  # 添加最后一个点
                    shadow_points.append((shadow_edges[-1][0][0] + np.cos(shadow_edges[-1][1]) * 1e6,
                    shadow_edges[-1][0][1] + np.sin(shadow_edges[-1][1]) * 1e6))
                    shadow_points.append((shadow_edges[0][0][0] + np.cos(shadow_edges[0][1]) * 1e6,
                    shadow_edges[0][0][1] + np.sin(shadow_edges[0][1]) * 1e6))

                    self.draw_polygon(shadow_points, filled=True, color=(0.5, 0.5, 0.5, 0.5))

    def add_geom(self, geom):
        self.geoms.append(geom)

    def add_onetime(self, geom):
        self.onetime_geoms.append(geom)

    def render(self, return_rgb_array=False):
        glClearColor(1, 1, 1, 1)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        self.transform.enable()
        for geom in self.onetime_geoms:
            geom.render()
        for geom in self.geoms:
            geom.render()

        self.transform.disable()
        arr = None
        if return_rgb_array:
            arr = pyglet.image.get_buffer_manager().get_color_buffer()
        self.window.flip()
        self.onetime_geoms = []
        return arr

    # Convenience
    def draw_circle(self, radius=10, res=30, filled=True, **attrs):
        geom = make_circle(radius=radius, res=res, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polygon(self, v, filled=True, **attrs):
        geom = make_polygon(v=v, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polyline(self, v, **attrs):
        geom = make_polyline(v=v)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_line(self, start, end, **attrs):
        geom = Line(start, end)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def get_array(self):
        self.window.flip()
        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        self.window.flip()
        arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
        arr = arr.reshape(self.height, self.width, 4)
        return arr[::-1, :, 0:3]


def _add_attrs(geom, attrs):
    if "color" in attrs:
        geom.set_color(*attrs["color"])
    if "linewidth" in attrs:
        geom.set_linewidth(attrs["linewidth"])


class Geom(object):
    def __init__(self):
        self._color = Color((0, 0, 0, 1.0))
        self.attrs = [self._color]

    def render(self):
        for attr in reversed(self.attrs):
            attr.enable()
        self.render1()
        for attr in self.attrs:
            attr.disable()

    def render1(self):
        raise NotImplementedError

    def add_attr(self, attr):
        self.attrs.append(attr)

    def set_color(self, r, g, b, alpha=1):
        self._color.vec4 = (r, g, b, alpha)


class Attr(object):
    def enable(self):
        raise NotImplementedError

    def disable(self):
        pass


class Transform(Attr):
    def __init__(self, translation=(0.0, 0.0), rotation=0.0, scale=(1, 1)):
        self.set_translation(*translation)
        self.set_rotation(rotation)
        self.set_scale(*scale)

    def enable(self):
        glPushMatrix()
        glTranslatef(self.translation[0], self.translation[1], 0)  # translate to GL loc ppint
        glRotatef(RAD2DEG * self.rotation, 0, 0, 1.0)
        glScalef(self.scale[0], self.scale[1], 1)

    def disable(self):
        glPopMatrix()

    def set_translation(self, newx, newy):
        self.translation = (float(newx), float(newy))

    def set_rotation(self, new):
        self.rotation = float(new)

    def set_scale(self, newx, newy):
        self.scale = (float(newx), float(newy))


class Color(Attr):
    def __init__(self, vec4):
        self.vec4 = vec4

    def enable(self):
        glColor4f(*self.vec4)


class LineStyle(Attr):
    def __init__(self, style):
        self.style = style

    def enable(self):
        glEnable(GL_LINE_STIPPLE)
        glLineStipple(1, self.style)

    def disable(self):
        glDisable(GL_LINE_STIPPLE)


class LineWidth(Attr):
    def __init__(self, stroke):
        self.stroke = stroke

    def enable(self):
        glLineWidth(self.stroke)


class Point(Geom):
    def __init__(self):
        Geom.__init__(self)

    def render1(self):
        glBegin(GL_POINTS)  # draw point
        glVertex3f(0.0, 0.0, 0.0)
        glEnd()


class FilledPolygon(Geom):
    def __init__(self, v):
        Geom.__init__(self)
        self.v = v

    def render1(self):
        if len(self.v) == 4:
            glBegin(GL_QUADS)
        elif len(self.v) > 4:
            glBegin(GL_POLYGON)
        else:
            glBegin(GL_TRIANGLES)
        for p in self.v:
            glVertex3f(p[0], p[1], 0)  # draw each vertex
        glEnd()

        color = (
            self._color.vec4[0] * 0.5, self._color.vec4[1] * 0.5, self._color.vec4[2] * 0.5, self._color.vec4[3] * 0.5)
        glColor4f(*color)
        glBegin(GL_LINE_LOOP)
        # for p in self.v:
        #   glVertex3f(p[0], p[1],0)  # draw each vertex
        glEnd()


def make_UAV(radius=10, res=30, filled=True):
    points1 = []
    points2 = []
    points3 = []
    points4 = []
    ang1 = (1 * radius, 0)
    ang2 = (0.3 * radius, 0.2 * radius)
    ang3 = (-0.2 * radius, 0.8 * radius)
    ang4 = (-0.3 * radius, 0.1 * radius)
    ang5 = (-0.31 * radius, 0.1 * radius)
    ang6 = (-0.8 * radius, 0.6 * radius)
    ang7 = (-0.8 * radius, -0.6 * radius)
    ang8 = (-0.31 * radius, -0.1 * radius)
    ang9 = (-0.3 * radius, -0.1 * radius)
    ang10 = (-0.2 * radius, -0.8 * radius)
    ang11 = (0.3 * radius, -0.2 * radius)
    ang12 = (-0.6 * radius, 0 * radius)

    points1.append(ang1)
    points1.append(ang2)
    points1.append(ang11)
    points2.append(ang2)
    points2.append(ang3)
    points2.append(ang4)
    points2.append(ang9)
    points2.append(ang10)
    points2.append(ang11)
    points3.append(ang4)
    points3.append(ang5)
    points3.append(ang8)
    points3.append(ang9)
    points4.append(ang5)
    points4.append(ang6)
    points4.append(ang12)
    points4.append(ang7)
    points4.append(ang8)
    return FilledPolygon(points1), FilledPolygon(points2), FilledPolygon(points3), FilledPolygon(points4)


def make_CAR(radius=10, res=30, filled=True):
    points1 = []  # 车身的四个角
    points2 = []  # 前左轮
    points3 = []  # 前右轮
    length = radius * 2
    width = radius
    wheel_radius = radius / 2
    # 定义车身四个角的坐标
    ang1 = (-length / 2, width / 2)  # 左前角
    ang2 = (length / 2, width / 2)  # 右前角
    ang3 = (length / 2, -width / 2)  # 右后角
    ang4 = (-length / 2, -width / 2)  # 左后角

    points1.append(ang1)
    points1.append(ang2)
    points1.append(ang3)
    points1.append(ang4)

    # 定义车轮的位置，车轮为圆形，前后各两个车轮
    wheel1_center = (-length / 2 + wheel_radius, width / 2 + wheel_radius)  # 前左轮
    wheel2_center = (-length / 2 + wheel_radius / 2, width / 2 + wheel_radius)  # 前右轮
    wheel3_center = (-length / 2 + wheel_radius, -width / 2 - wheel_radius)  # 后左轮
    wheel4_center = (-length / 2 + wheel_radius / 2, -width / 2 - wheel_radius)  # 后右轮

    # 定义车轮的位置，车轮为圆形，前后各两个车轮
    wheel5_center = (length / 2 - wheel_radius, width / 2 + wheel_radius)  # 前左轮
    wheel6_center = (length / 2 - wheel_radius / 2, width / 2 + wheel_radius)  # 前右轮
    wheel7_center = (length / 2 - wheel_radius, -width / 2 - wheel_radius)  # 后左轮
    wheel8_center = (length / 2 - wheel_radius / 2, -width / 2 - wheel_radius)  # 后右轮

    # 车轮（使用圆形来表示）
    points2.append(wheel1_center)
    points2.append(wheel2_center)
    points2.append(wheel3_center)
    points2.append(wheel4_center)

    points3.append(wheel5_center)
    points3.append(wheel6_center)
    points3.append(wheel7_center)
    points3.append(wheel8_center)

    return FilledPolygon(points1), FilledPolygon(points2), FilledPolygon(points3)


def make_circle(radius=10, res=30, filled=True):
    points = []
    for i in range(res):
        ang = 2 * math.pi * i / res
        points.append((math.cos(ang) * radius, math.sin(ang) * radius))
    if filled:
        return FilledPolygon(points)
    else:
        return PolyLine(points, True)


def make_sector(radius=10, theta=math.pi, res=60, filled=True):
    points = []
    for i in range(-res // 2, res // 2):
        ang = theta * i / res
        points.append((math.cos(ang) * radius, math.sin(ang) * radius))
    points.append((0, 0))
    if filled:
        return FilledPolygon(points)
    else:
        return PolyLine(points, True)


def make_rectangle(width=10, height=5, filled=True):
    # Define the four corners of the rectangle
    points = [
        (-width/2, -height/2),  # Bottom-left corner
        (width/2, -height/2),  # Bottom-right corner
        (width/2, height/2),  # Top-right corner
        (-width/2, height/2)  # Top-left corner
    ]

    if filled:
        return FilledPolygon(points)  # Assuming FilledPolygon is defined elsewhere
    else:
        return PolyLine(points, True)  # Assuming PolyLine is defined elsewhere
def make_polygon(v, filled=True):
    if filled:
        return FilledPolygon(v)
    else:
        return PolyLine(v, True)


def make_polyline(v):
    return PolyLine(v, False)


def make_capsule(length, width):
    l, r, t, b = 0, length, width / 2, -width / 2
    box = make_polygon([(l, b), (l, t), (r, t), (r, b)])
    circ0 = make_circle(width / 2)
    circ1 = make_circle(width / 2)
    circ1.add_attr(Transform(translation=(length, 0)))
    geom = Compound([box, circ0, circ1])
    return geom


class Compound(Geom):
    def __init__(self, gs):
        Geom.__init__(self)
        self.gs = gs
        for g in self.gs:
            g.attrs = [a for a in g.attrs if not isinstance(a, Color)]

    def render1(self):
        for g in self.gs:
            g.render()


class PolyLine(Geom):
    def __init__(self, v, close):
        Geom.__init__(self)
        self.v = v
        self.close = close
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)

    def render1(self):
        glBegin(GL_LINE_LOOP if self.close else GL_LINE_STRIP)
        for p in self.v:
            glVertex3f(p[0], p[1], 0)  # draw each vertex
        glEnd()

    def set_linewidth(self, x):
        self.linewidth.stroke = x


class Line(Geom):
    def __init__(self, start=(0.0, 0.0), end=(0.0, 0.0)):
        Geom.__init__(self)
        self.start = start
        self.end = end
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)

    def render1(self):
        glBegin(GL_LINES)
        glVertex2f(*self.start)
        glVertex2f(*self.end)
        glEnd()


class Image(Geom):
    def __init__(self, fname, width, height):
        Geom.__init__(self)
        self.width = width
        self.height = height
        img = pyglet.image.load(fname)
        self.img = img
        self.flip = False

    def render1(self):
        self.img.blit(-self.width / 2, -self.height / 2, width=self.width, height=self.height)
