import taichi as ti

ti.init(arch=ti.gpu)

res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

Ka = ti.field(dtype=ti.f32, shape=())
Kd = ti.field(dtype=ti.f32, shape=())
Ks = ti.field(dtype=ti.f32, shape=())
shininess = ti.field(dtype=ti.f32, shape=())

@ti.func
def normalize(v):
    return v / v.norm(1e-6)

@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N

# ---- 球体 ----
@ti.func
def intersect_sphere(ro, rd, center, radius):
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])

    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c

    if delta > 0:
        t1 = (-b - ti.sqrt(delta)) / 2.0
        if t1 > 0:
            t = t1
            p = ro + rd * t
            normal = normalize(p - center)

    return t, normal


# ---- 圆锥 ----
@ti.func
def intersect_cone(ro, rd, apex, base_y, radius):
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])

    H = apex.y - base_y
    k = (radius / H) ** 2

    ro_local = ro - apex

    A = rd.x**2 + rd.z**2 - k * rd.y**2
    B = 2.0 * (ro_local.x * rd.x + ro_local.z * rd.z - k * ro_local.y * rd.y)
    C = ro_local.x**2 + ro_local.z**2 - k * ro_local.y**2

    if ti.abs(A) > 1e-5:
        delta = B**2 - 4.0 * A * C
        if delta > 0:
            t1 = (-B - ti.sqrt(delta)) / (2.0 * A)
            t2 = (-B + ti.sqrt(delta)) / (2.0 * A)

            t_first = t1 if t1 < t2 else t2
            t_second = t2 if t1 < t2 else t1

            y1 = ro_local.y + t_first * rd.y
            if t_first > 0 and -H <= y1 <= 0:
                t = t_first
            else:
                y2 = ro_local.y + t_second * rd.y
                if t_second > 0 and -H <= y2 <= 0:
                    t = t_second

            if t > 0:
                p_local = ro_local + rd * t
                normal = normalize(ti.Vector([p_local.x, -k * p_local.y, p_local.z]))

    return t, normal

@ti.kernel
def render():
    for i, j in pixels:

        u = (i - res_x * 0.5) / res_y * 2.0
        v = (j - res_y * 0.5) / res_y * 2.0

        ro = ti.Vector([0.0, 0.0, 5.0])   # 摄像机
        rd = normalize(ti.Vector([u, v, -1.0]))

        min_t = 1e10
        hit_normal = ti.Vector([0.0, 0.0, 0.0])
        hit_color = ti.Vector([0.0, 0.0, 0.0])

        # ---- 球体 ----
        t_sph, n_sph = intersect_sphere(
            ro, rd,
            ti.Vector([-1.2, -0.2, 0.0]),
            1.2
        )
        if 0 < t_sph < min_t:
            min_t = t_sph
            hit_normal = n_sph
            hit_color = ti.Vector([0.8, 0.1, 0.1])

        # ---- 圆锥 ----
        t_cone, n_cone = intersect_cone(
            ro, rd,
            ti.Vector([1.2, 1.2, 0.0]),
            -1.4,
            1.2
        )
        if 0 < t_cone < min_t:
            min_t = t_cone
            hit_normal = n_cone
            hit_color = ti.Vector([0.6, 0.2, 0.8])

        color = ti.Vector([0.95,0.95, 0.95])

        if min_t < 1e9:
            p = ro + rd * min_t
            N = hit_normal

            light_pos = ti.Vector([2.0, 3.0, 4.0])
            light_color = ti.Vector([1.0, 1.0, 1.0])

            L = normalize(light_pos - p)
            V = normalize(ro - p)
            R = normalize(reflect(-L, N))

            ambient = Ka[None] * light_color * hit_color

            diff = ti.max(0.0, N.dot(L))
            diffuse = Kd[None] * diff * light_color * hit_color

            spec = ti.max(0.0, R.dot(V)) ** shininess[None]
            specular = Ks[None] * spec * light_color

            color = ambient + diffuse + specular

        pixels[i, j] = ti.math.clamp(color, 0.0, 1.0)

def main():
    window = ti.ui.Window("Phong Lighting Experiment", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()

    Ka[None] = 0.2
    Kd[None] = 0.7
    Ks[None] = 0.5
    shininess[None] = 32.0

    while window.running:

        render()
        canvas.set_image(pixels)

        with gui.sub_window("Phong Parameters", 0.7, 0.05, 0.28, 0.25):
            Ka[None] = gui.slider_float("Ka (Ambient)", Ka[None], 0.0, 1.0)
            Kd[None] = gui.slider_float("Kd (Diffuse)", Kd[None], 0.0, 1.0)
            Ks[None] = gui.slider_float("Ks (Specular)", Ks[None], 0.0, 1.0)
            shininess[None] = gui.slider_float("Shininess", shininess[None], 1.0, 128.0)

        window.show()


if __name__ == "__main__":
    main()