import taichi as ti
import math

ti.init(arch=ti.cpu)

vertices = ti.Vector.field(3, dtype=ti.f32, shape=8)
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=8)

edges = [
    (0,1),(1,2),(2,3),(3,0),  
    (4,5),(5,6),(6,7),(7,4),  
    (0,4),(1,5),(2,6),(3,7)   
]

@ti.func
def get_model_matrix(angle):
    rad = angle * math.pi / 180.0
    c = ti.cos(rad)
    s = ti.sin(rad)

    return ti.Matrix([
        [ c, 0.0, s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_view_matrix(eye_pos):
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_projection_matrix(fov, aspect, zNear, zFar):
    n = -zNear
    f = -zFar

    fov_rad = fov * math.pi / 180.0
    t = ti.tan(fov_rad / 2.0) * ti.abs(n)
    r = aspect * t

    # 透视投影
    return ti.Matrix([
        [n/r, 0,   0,              0],
        [0,   n/t, 0,              0],
        [0,   0,  (n+f)/(n-f), 2*n*f/(f-n)],
        [0,   0,  -1,              0]
    ])

@ti.kernel
def compute_transform(angle: float):
    eye_pos = ti.Vector([0.0, 0.0, 5.0])

    model = get_model_matrix(angle)
    view = get_view_matrix(eye_pos)
    proj = get_projection_matrix(45.0, 1.0, 0.1, 50.0)

    mvp = proj @ view @ model

    for i in range(8):
        v = vertices[i]
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])

        v_clip = mvp @ v4
        v_ndc = v_clip / v_clip[3]

        screen_coords[i][0] = (v_ndc[0] + 1.0) / 2.0
        screen_coords[i][1] = (v_ndc[1] + 1.0) / 2.0

def main():
    vertices[0] = [-1, -1, -1]
    vertices[1] = [ 1, -1, -1]
    vertices[2] = [ 1,  1, -1]
    vertices[3] = [-1,  1, -1]
    vertices[4] = [-1, -1,  1]
    vertices[5] = [ 1, -1,  1]
    vertices[6] = [ 1,  1,  1]
    vertices[7] = [-1,  1,  1]

    gui = ti.GUI("3D Cube Rotation", res=(700, 700))

    angle = 0.0

    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == 'a':
                angle += 10
            elif gui.event.key == 'd':
                angle -= 10
            elif gui.event.key == ti.GUI.ESCAPE:
                break

        compute_transform(angle)

        for e in edges:
            a = screen_coords[e[0]]
            b = screen_coords[e[1]]
            gui.line(a, b, radius=2, color=0x00FFFF)

        gui.show()

if __name__ == '__main__':
    main()