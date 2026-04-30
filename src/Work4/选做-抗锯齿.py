import taichi as ti
import math

ti.init(arch=ti.gpu)

WIDTH, HEIGHT = 960, 540

MAX_BOUNCES_LIMIT = 5
MAX_SAMPLES_LIMIT = 16

EPS = 1e-4
FOV = math.radians(45.0)

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))


@ti.func
def safe_normalize(v):
    return v / ti.sqrt(v.dot(v) + 1e-8)


@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N


@ti.func
def sphere_intersect(ro, rd, center, radius):
    oc = ro - center

    a = rd.dot(rd)
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius

    delta = b * b - 4.0 * a * c

    hit = False
    t = 1e20

    if delta > 0.0:
        sqrtd = ti.sqrt(delta)

        t1 = (-b - sqrtd) / (2.0 * a)
        t2 = (-b + sqrtd) / (2.0 * a)

        if t1 > EPS:
            t = t1
            hit = True
        elif t2 > EPS:
            t = t2
            hit = True

    return hit, t


@ti.func
def plane_intersect(ro, rd):
    hit = False
    t = 1e20

    if ti.abs(rd.y) > 1e-6:
        temp_t = (-1.0 - ro.y) / rd.y

        if temp_t > EPS:
            hit = True
            t = temp_t

    return hit, t


@ti.func
def scene_intersect(ro, rd):
    closest_t = 1e20

    hit_pos = ti.Vector([0.0, 0.0, 0.0])
    hit_normal = ti.Vector([0.0, 1.0, 0.0])

    material_id = 0

    red_sphere_center = ti.Vector([-1.5, 0.0, 0.0])
    mirror_sphere_center = ti.Vector([1.5, 0.0, 0.0])

    hit_red, t_red = sphere_intersect(
        ro,
        rd,
        red_sphere_center,
        1.0
    )

    if hit_red and t_red < closest_t:
        closest_t = t_red
        hit_pos = ro + rd * closest_t
        hit_normal = safe_normalize(hit_pos - red_sphere_center)
        material_id = 1

    hit_mirror, t_mirror = sphere_intersect(
        ro,
        rd,
        mirror_sphere_center,
        1.0
    )

    if hit_mirror and t_mirror < closest_t:
        closest_t = t_mirror
        hit_pos = ro + rd * closest_t
        hit_normal = safe_normalize(hit_pos - mirror_sphere_center)
        material_id = 2

    hit_plane, t_plane = plane_intersect(ro, rd)

    if hit_plane and t_plane < closest_t:
        closest_t = t_plane
        hit_pos = ro + rd * closest_t
        hit_normal = ti.Vector([0.0, 1.0, 0.0])
        material_id = 3

    return material_id, closest_t, hit_pos, hit_normal


@ti.func
def checker_color(pos):
    ix = ti.floor(pos.x)
    iz = ti.floor(pos.z)

    checker = ti.cast(ix + iz, ti.i32) & 1

    color = ti.Vector([0.85, 0.85, 0.85])

    if checker == 1:
        color = ti.Vector([0.08, 0.08, 0.08])

    return color


@ti.func
def background_color(rd):
    t = 0.5 * (rd.y + 1.0)

    bottom = ti.Vector([0.75, 0.82, 0.95])
    top = ti.Vector([0.35, 0.55, 0.90])

    return (1.0 - t) * bottom + t * top


@ti.func
def in_shadow(point, normal, light_pos):
    shadow_origin = point + normal * EPS

    light_vec = light_pos - shadow_origin
    light_dist = ti.sqrt(light_vec.dot(light_vec))
    light_dir = light_vec / light_dist

    mat, t, _, _ = scene_intersect(shadow_origin, light_dir)

    shadow = False

    if mat != 0 and t < light_dist:
        shadow = True

    return shadow


@ti.func
def shade_diffuse(point, normal, view_dir, base_color, light_pos):
    ambient = 0.12 * base_color

    light_dir = safe_normalize(light_pos - point)

    diff = ti.max(normal.dot(light_dir), 0.0)
    diffuse = base_color * diff

    half_dir = safe_normalize(light_dir + view_dir)
    spec = ti.pow(ti.max(normal.dot(half_dir), 0.0), 64.0)
    specular = ti.Vector([1.0, 1.0, 1.0]) * spec * 0.25

    color = ambient

    if not in_shadow(point, normal, light_pos):
        color += diffuse + specular

    return color


@ti.func
def trace_ray(ray_origin, ray_dir, light_pos, max_bounces):
    final_color = ti.Vector([0.0, 0.0, 0.0])
    throughput = ti.Vector([1.0, 1.0, 1.0])

    for bounce in range(MAX_BOUNCES_LIMIT):
        if bounce < max_bounces:
            mat, t, hit_pos, normal = scene_intersect(ray_origin, ray_dir)

            if mat == 0:
                final_color += throughput * background_color(ray_dir)
                break

            view_dir = safe_normalize(-ray_dir)

            if mat == 1:
                base_color = ti.Vector([0.95, 0.12, 0.08])
                color = shade_diffuse(hit_pos, normal, view_dir, base_color, light_pos)

                final_color += throughput * color
                break

            elif mat == 2:
                new_dir = safe_normalize(reflect(ray_dir, normal))

                ray_origin = hit_pos + normal * EPS
                ray_dir = new_dir

                throughput *= ti.Vector([0.82, 0.82, 0.82])

            elif mat == 3:
                base_color = checker_color(hit_pos)
                color = shade_diffuse(hit_pos, normal, view_dir, base_color, light_pos)

                final_color += throughput * color
                break

    return final_color


@ti.kernel
def render(
    light_x: float,
    light_y: float,
    light_z: float,
    max_bounces: int,
    samples_per_pixel: int
):
    light_pos = ti.Vector([light_x, light_y, light_z])

    camera_pos = ti.Vector([0.0, 0.3, 5.0])
    target = ti.Vector([0.0, -0.1, 0.0])

    forward = safe_normalize(target - camera_pos)
    world_up = ti.Vector([0.0, 1.0, 0.0])

    right = safe_normalize(forward.cross(world_up))
    up = safe_normalize(right.cross(forward))

    aspect = WIDTH / HEIGHT
    scale = ti.tan(FOV * 0.5)

    for i, j in pixels:
        color_sum = ti.Vector([0.0, 0.0, 0.0])

        used_samples = 0

        for sample_id in range(MAX_SAMPLES_LIMIT):
            if sample_id < samples_per_pixel:
                # MSAA 核心：
                # 在当前像素内部随机生成一个采样点，而不是固定采样像素中心。
                jitter_x = ti.random(ti.f32)
                jitter_y = ti.random(ti.f32)

                u = (2.0 * (i + jitter_x) / WIDTH - 1.0) * aspect * scale
                v = (2.0 * (j + jitter_y) / HEIGHT - 1.0) * scale

                ray_origin = camera_pos
                ray_dir = safe_normalize(forward + right * u + up * v)

                color_sum += trace_ray(ray_origin, ray_dir, light_pos, max_bounces)
                used_samples += 1

        final_color = color_sum / ti.cast(used_samples, ti.f32)

        final_color = ti.min(final_color, ti.Vector([1.0, 1.0, 1.0]))
        final_color = ti.sqrt(final_color)

        pixels[i, j] = final_color


def main():
    light_x = 0.0
    light_y = 5.0
    light_z = 3.0

    max_bounces = 3
    samples_per_pixel = 4

    window = ti.ui.Window(
        "Whitted Ray Tracing - MSAA Anti-Aliasing",
        (WIDTH, HEIGHT),
        vsync=True
    )

    canvas = window.get_canvas()
    gui = window.get_gui()

    while window.running:
        with gui.sub_window("Control Panel", 0.02, 0.02, 0.33, 0.34):
            gui.text("Light Position")
            light_x = gui.slider_float("Light X", light_x, -6.0, 6.0)
            light_y = gui.slider_float("Light Y", light_y, 0.5, 8.0)
            light_z = gui.slider_float("Light Z", light_z, -6.0, 6.0)

            gui.text("Ray Tracing")
            max_bounces = gui.slider_int("Max Bounces", max_bounces, 1, 5)

            gui.text("Anti-Aliasing")
            samples_per_pixel = gui.slider_int(
                "Samples Per Pixel",
                samples_per_pixel,
                1,
                16
            )

            gui.text("1 sample: jagged edges")
            gui.text("4-16 samples: smoother edges")

        render(
            light_x,
            light_y,
            light_z,
            max_bounces,
            samples_per_pixel
        )

        canvas.set_image(pixels)
        window.show()


if __name__ == "__main__":
    main()