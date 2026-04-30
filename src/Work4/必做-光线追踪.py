import taichi as ti
import math

ti.init(arch=ti.gpu)

WIDTH, HEIGHT = 960, 540
MAX_BOUNCES_LIMIT = 5

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))


@ti.func
def normalize(v):
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

    t = 1e20
    hit = False

    if delta > 0.0:
        sqrtd = ti.sqrt(delta)
        t1 = (-b - sqrtd) / (2.0 * a)
        t2 = (-b + sqrtd) / (2.0 * a)

        if t1 > 1e-4:
            t = t1
            hit = True
        elif t2 > 1e-4:
            t = t2
            hit = True

    return hit, t


@ti.func
def plane_intersect(ro, rd):
    t = 1e20
    hit = False

    if ti.abs(rd.y) > 1e-5:
        temp_t = (-1.0 - ro.y) / rd.y
        if temp_t > 1e-4:
            t = temp_t
            hit = True

    return hit, t


@ti.func
def scene_intersect(ro, rd):
    closest_t = 1e20
    hit_pos = ti.Vector([0.0, 0.0, 0.0])
    hit_normal = ti.Vector([0.0, 1.0, 0.0])
    material_id = 0

    hit1, t1 = sphere_intersect(
        ro, rd,
        ti.Vector([-1.5, 0.0, 0.0]),
        1.0
    )

    if hit1 and t1 < closest_t:
        closest_t = t1
        hit_pos = ro + rd * closest_t
        hit_normal = normalize(hit_pos - ti.Vector([-1.5, 0.0, 0.0]))
        material_id = 1

    hit2, t2 = sphere_intersect(
        ro, rd,
        ti.Vector([1.5, 0.0, 0.0]),
        1.0
    )

    if hit2 and t2 < closest_t:
        closest_t = t2
        hit_pos = ro + rd * closest_t
        hit_normal = normalize(hit_pos - ti.Vector([1.5, 0.0, 0.0]))
        material_id = 2

    hit3, t3 = plane_intersect(ro, rd)

    if hit3 and t3 < closest_t:
        closest_t = t3
        hit_pos = ro + rd * closest_t
        hit_normal = ti.Vector([0.0, 1.0, 0.0])
        material_id = 3

    return material_id, closest_t, hit_pos, hit_normal


@ti.func
def checker_color(pos):
    ix = ti.floor(pos.x)
    iz = ti.floor(pos.z)
    checker = (ti.cast(ix + iz, ti.i32) & 1)

    color = ti.Vector([0.85, 0.85, 0.85])
    if checker == 1:
        color = ti.Vector([0.08, 0.08, 0.08])

    return color


@ti.func
def background_color(rd):
    t = 0.5 * (rd.y + 1.0)
    return (1.0 - t) * ti.Vector([0.75, 0.82, 0.95]) + t * ti.Vector([0.35, 0.55, 0.90])


@ti.func
def in_shadow(point, normal, light_pos):
    shadow_origin = point + normal * 1e-4
    light_dir = light_pos - shadow_origin
    light_dist = ti.sqrt(light_dir.dot(light_dir))
    light_dir = light_dir / light_dist

    mat, t, _, _ = scene_intersect(shadow_origin, light_dir)

    shadow = False
    if mat != 0 and t < light_dist:
        shadow = True

    return shadow


@ti.func
def shade_diffuse(point, normal, view_dir, base_color, light_pos):
    ambient = 0.12 * base_color

    light_dir = normalize(light_pos - point)
    diff = ti.max(normal.dot(light_dir), 0.0)

    diffuse = base_color * diff

    half_dir = normalize(light_dir + view_dir)
    spec = ti.pow(ti.max(normal.dot(half_dir), 0.0), 64.0)
    specular = ti.Vector([1.0, 1.0, 1.0]) * spec * 0.25

    color = ambient

    if not in_shadow(point, normal, light_pos):
        color += diffuse + specular

    return color


@ti.kernel
def render(light_x: float, light_y: float, light_z: float, max_bounces: float):
    light_pos = ti.Vector([light_x, light_y, light_z])

    camera_pos = ti.Vector([0.0, 0.3, 5.0])
    target = ti.Vector([0.0, -0.1, 0.0])

    forward = normalize(target - camera_pos)
    world_up = ti.Vector([0.0, 1.0, 0.0])
    right = normalize(forward.cross(world_up))
    up = normalize(right.cross(forward))

    aspect = WIDTH / HEIGHT
    fov = 45.0 * math.pi / 180.0
    scale = ti.tan(fov * 0.5)

    for i, j in pixels:
        u = (2.0 * (i + 0.5) / WIDTH - 1.0) * aspect * scale
        v = (2.0 * (j + 0.5) / HEIGHT - 1.0) * scale

        ray_origin = camera_pos
        ray_dir = normalize(forward + right * u + up * v)

        final_color = ti.Vector([0.0, 0.0, 0.0])
        throughput = ti.Vector([1.0, 1.0, 1.0])

        for bounce in range(MAX_BOUNCES_LIMIT):
            if bounce < max_bounces:
                mat, t, hit_pos, normal = scene_intersect(ray_origin, ray_dir)

                if mat == 0:
                    final_color += throughput * background_color(ray_dir)
                    break

                view_dir = normalize(-ray_dir)

                if mat == 1:
                    base_color = ti.Vector([0.95, 0.12, 0.08])
                    color = shade_diffuse(hit_pos, normal, view_dir, base_color, light_pos)
                    final_color += throughput * color
                    break

                elif mat == 2:
                    ray_origin = hit_pos + normal * 1e-4
                    ray_dir = normalize(reflect(ray_dir, normal))
                    throughput *= ti.Vector([0.82, 0.82, 0.82])

                elif mat == 3:
                    base_color = checker_color(hit_pos)
                    color = shade_diffuse(hit_pos, normal, view_dir, base_color, light_pos)
                    final_color += throughput * color
                    break

        final_color = ti.sqrt(ti.min(final_color, ti.Vector([1.0, 1.0, 1.0])))
        pixels[i, j] = final_color


def main():
    light_x = 0.0
    light_y = 5.0
    light_z = 3.0
    max_bounces = 3

    window = ti.ui.Window(
        "Whitted Style Ray Tracing - Taichi",
        (WIDTH, HEIGHT),
        vsync=True
    )

    canvas = window.get_canvas()
    gui = window.get_gui()

    while window.running:
        with gui.sub_window("Control Panel", 0.02, 0.02, 0.30, 0.28):
            gui.text("Light Position")
            light_x = gui.slider_float("Light X", light_x, -6.0, 6.0)
            light_y = gui.slider_float("Light Y", light_y, 0.5, 8.0)
            light_z = gui.slider_float("Light Z", light_z, -6.0, 6.0)

            gui.text("Ray Tracing")
            max_bounces = gui.slider_int("Max Bounces", max_bounces, 1, 5)

            gui.text("Objects:")
            gui.text("Red Sphere: Diffuse")
            gui.text("Silver Sphere: Mirror")
            gui.text("Ground: Checkerboard")

        render(light_x, light_y, light_z, max_bounces)

        canvas.set_image(pixels)
        window.show()


if __name__ == "__main__":
    main()