import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

WIDTH, HEIGHT = 800, 800
MAX_CONTROL_POINTS = 100
NUM_SEGMENTS = 1000

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)

gui_indices = ti.field(dtype=ti.i32, shape=MAX_CONTROL_POINTS * 2)

curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_SEGMENTS + 1)

# De Casteljau 算法（递归实现）
def de_casteljau(points, t):
    if len(points) == 1:
        return points[0]

    new_points = []
    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i + 1]

        x = (1 - t) * p0[0] + t * p1[0]
        y = (1 - t) * p0[1] + t * p1[1]

        new_points.append([x, y])

    return de_casteljau(new_points, t)

# 清屏
@ti.kernel
def clear_pixels():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])

# 绘制贝塞尔曲线
@ti.kernel
def draw_curve(n: int):
    for i in range(n):
        p = curve_points_field[i]

        x = ti.cast(p[0] * WIDTH, ti.i32)
        y = ti.cast(p[1] * HEIGHT, ti.i32)

        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            pixels[x, y] = ti.Vector([0.0, 1.0, 0.0])

def main():
    window = ti.ui.Window("Bezier Curve Demo", (WIDTH, HEIGHT))
    canvas = window.get_canvas()

    control_points = []

    while window.running:

        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.LMB:
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = window.get_cursor_pos()
                    control_points.append(pos)

            elif e.key == 'c':
                control_points = []

        clear_pixels()

        # 计算贝塞尔曲线
        n = len(control_points)
        if n >= 2:
            curve_np = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)

            for i in range(NUM_SEGMENTS + 1):
                t = i / NUM_SEGMENTS
                curve_np[i] = de_casteljau(control_points, t)

            curve_points_field.from_numpy(curve_np)

            draw_curve(NUM_SEGMENTS + 1)

        canvas.set_image(pixels)

        if n > 0:
            np_points = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
            np_points[:n] = np.array(control_points, dtype=np.float32)

            gui_points.from_numpy(np_points)

            canvas.circles(
                gui_points,
                radius=0.008,   
                color=(1.0, 0.0, 0.0)
            )

            if n >= 2:
                indices = []
                for i in range(n - 1):
                    indices.append(i)
                    indices.append(i + 1)

                np_indices = np.zeros(MAX_CONTROL_POINTS * 2, dtype=np.int32)
                np_indices[:len(indices)] = np.array(indices, dtype=np.int32)

                gui_indices.from_numpy(np_indices)

                canvas.lines(
                    gui_points,
                    width=0.003,  
                    indices=gui_indices,
                    color=(0.5, 0.5, 0.5)
                )

        window.show()

if __name__ == "__main__":
    main()