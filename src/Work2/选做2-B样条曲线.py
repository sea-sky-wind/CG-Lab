import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

WIDTH, HEIGHT = 800, 800
MAX_CONTROL_POINTS = 100
NUM_SEGMENTS = 1000

# GPU
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
gui_indices = ti.field(dtype=ti.i32, shape=MAX_CONTROL_POINTS * 2)
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_SEGMENTS * 2)

# De Casteljau（Bezier）
def de_casteljau(points, t):
    if len(points) == 1:
        return points[0]
    new_pts = []
    for i in range(len(points) - 1):
        p0, p1 = points[i], points[i + 1]
        new_pts.append([
            (1 - t) * p0[0] + t * p1[0],
            (1 - t) * p0[1] + t * p1[1]
        ])
    return de_casteljau(new_pts, t)

# 三次均匀B样条（矩阵法）
def bspline_segment(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t

    # 三次B样条基函数
    b0 = (-t3 + 3*t2 - 3*t + 1) / 6.0
    b1 = (3*t3 - 6*t2 + 4) / 6.0
    b2 = (-3*t3 + 3*t2 + 3*t + 1) / 6.0
    b3 = t3 / 6.0

    x = b0*p0[0] + b1*p1[0] + b2*p2[0] + b3*p3[0]
    y = b0*p0[1] + b1*p1[1] + b2*p2[1] + b3*p3[1]

    return [x, y]

@ti.kernel
def clear_pixels():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def draw_curve(n: int):
    for i in range(n):
        p = curve_points_field[i]
        x = ti.cast(p[0] * WIDTH, ti.i32)
        y = ti.cast(p[1] * HEIGHT, ti.i32)

        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            pixels[x, y] = ti.Vector([0.0, 1.0, 0.0])

def main():
    window = ti.ui.Window("Bezier / B-Spline", (WIDTH, HEIGHT))
    canvas = window.get_canvas()

    control_points = []
    mode = 0  # 0=Bezier, 1=B-spline

    while window.running:

        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.LMB:
                if len(control_points) < MAX_CONTROL_POINTS:
                    control_points.append(window.get_cursor_pos())

            elif e.key == 'c':
                control_points = []

            elif e.key == 'b':
                mode = 1 - mode
                print("Mode:", "B-Spline" if mode else "Bezier")

        clear_pixels()

        n = len(control_points)
        curve_points = []

        # Bezier
        if mode == 0 and n >= 2:
            for i in range(NUM_SEGMENTS):
                t = i / NUM_SEGMENTS
                curve_points.append(de_casteljau(control_points, t))

        # B样条（三次均匀）
        elif mode == 1 and n >= 4:
            for i in range(n - 3):
                p0, p1, p2, p3 = control_points[i:i+4]
                for j in range(NUM_SEGMENTS // (n - 3)):
                    t = j / (NUM_SEGMENTS // (n - 3))
                    curve_points.append(bspline_segment(p0, p1, p2, p3, t))

        if len(curve_points) > 0:
            curve_np = np.zeros((len(curve_points), 2), dtype=np.float32)
            curve_np[:len(curve_points)] = np.array(curve_points, dtype=np.float32)

            curve_points_field.from_numpy(curve_np)
            draw_curve(len(curve_points))

        canvas.set_image(pixels)

        if n > 0:
            np_points = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
            np_points[:n] = np.array(control_points, dtype=np.float32)
            gui_points.from_numpy(np_points)

            canvas.circles(gui_points, radius=0.008, color=(1, 0, 0))

            if n >= 2:
                indices = []
                for i in range(n - 1):
                    indices.extend([i, i + 1])

                np_idx = np.zeros(MAX_CONTROL_POINTS * 2, dtype=np.int32)
                np_idx[:len(indices)] = np.array(indices, dtype=np.int32)
                gui_indices.from_numpy(np_idx)

                canvas.lines(gui_points, width=0.003, indices=gui_indices, color=(0.5, 0.5, 0.5))

        window.show()

if __name__ == "__main__":
    main()