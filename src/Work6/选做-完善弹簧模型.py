import taichi as ti

ti.init(arch=ti.gpu)

N = 20
mass = 1.0
dt = 5e-4
k_s = 10000.0       
k_shear = 7000.0    
k_bend = 3000.0   
k_d = 1.0
gravity = ti.Vector([0.0, -9.8, 0.0])
max_velocity = 50.0

x = ti.Vector.field(3, dtype=float, shape=N * N)
v = ti.Vector.field(3, dtype=float, shape=N * N)
f = ti.Vector.field(3, dtype=float, shape=N * N)
is_fixed = ti.field(dtype=int, shape=N * N)

x_next = ti.Vector.field(3, dtype=float, shape=N * N)
v_next = ti.Vector.field(3, dtype=float, shape=N * N)
f_next = ti.Vector.field(3, dtype=float, shape=N * N)

max_springs = N * N * 8
spring_indices = ti.field(dtype=int, shape=max_springs * 2)
spring_pairs = ti.Vector.field(2, dtype=int, shape=max_springs)
spring_lengths = ti.field(dtype=float, shape=max_springs)
spring_ks = ti.field(dtype=float, shape=max_springs)   
num_springs = ti.field(dtype=int, shape=())

@ti.kernel
def init_positions():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        x[idx] = ti.Vector([i * 0.05 - 0.5, 0.8, j * 0.05 - 0.5])
        v[idx] = ti.Vector([0.0, 0.0, 0.0])
        f[idx] = ti.Vector([0.0, 0.0, 0.0])
        if j == 0 and (i == 0 or i == N - 1):
            is_fixed[idx] = 1
        else:
            is_fixed[idx] = 0

@ti.kernel
def init_springs():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j

        if i < N - 1:
            nb = (i + 1) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c]   = ti.Vector([idx, nb])
            spring_lengths[c] = (x[idx] - x[nb]).norm()
            spring_ks[c]      = k_s

        if j < N - 1:
            nb = i * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c]   = ti.Vector([idx, nb])
            spring_lengths[c] = (x[idx] - x[nb]).norm()
            spring_ks[c]      = k_s

        if i < N - 1 and j < N - 1:
            nb = (i + 1) * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c]   = ti.Vector([idx, nb])
            spring_lengths[c] = (x[idx] - x[nb]).norm()
            spring_ks[c]      = k_shear

        if i < N - 1 and j > 0:
            nb = (i + 1) * N + (j - 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c]   = ti.Vector([idx, nb])
            spring_lengths[c] = (x[idx] - x[nb]).norm()
            spring_ks[c]      = k_shear

        if i < N - 2:
            nb = (i + 2) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c]   = ti.Vector([idx, nb])
            spring_lengths[c] = (x[idx] - x[nb]).norm()
            spring_ks[c]      = k_bend

        if j < N - 2:
            nb = i * N + (j + 2)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c]   = ti.Vector([idx, nb])
            spring_lengths[c] = (x[idx] - x[nb]).norm()
            spring_ks[c]      = k_bend

@ti.kernel
def init_spring_indices():
    for i in range(num_springs[None]):
        spring_indices[i * 2]     = spring_pairs[i][0]
        spring_indices[i * 2 + 1] = spring_pairs[i][1]

def init_cloth():
    num_springs[None] = 0
    init_positions()
    init_springs()
    init_spring_indices()


@ti.func
def compute_forces_on(pos, vel, force):
    for i in range(N * N):
        force[i] = gravity * mass - k_d * vel[i]
    for s in range(num_springs[None]):
        idx_a = spring_pairs[s][0]
        idx_b = spring_pairs[s][1]
        d     = pos[idx_a] - pos[idx_b]
        dist  = d.norm()
        if dist > 1e-6:
            f_spring = -spring_ks[s] * (dist - spring_lengths[s]) * (d / dist)
            ti.atomic_add(force[idx_a],  f_spring)
            ti.atomic_add(force[idx_b], -f_spring)

@ti.func
def clamp_velocity(vel, idx):
    vel_norm = vel[idx].norm()
    if vel_norm > max_velocity:
        vel[idx] = vel[idx] / vel_norm * max_velocity

@ti.kernel
def step_explicit():
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            x[i] += v[i] * dt
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)

@ti.kernel
def step_semi_implicit():
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
            x[i] += v[i] * dt

@ti.kernel
def step_implicit_iter():
    for i in range(N * N):
        v_next[i] = v[i]
        x_next[i] = x[i]
    for _ in ti.static(range(3)):
        compute_forces_on(x_next, v_next, f_next)
        for i in range(N * N):
            if is_fixed[i] == 0:
                v_next[i] = v[i] + (f_next[i] / mass) * dt
                clamp_velocity(v_next, i)
                x_next[i] = x[i] + v_next[i] * dt
    for i in range(N * N):
        v[i] = v_next[i]
        x[i] = x_next[i]

def main():
    init_cloth()

    window = ti.ui.Window("Mass-Spring Cloth - Full Springs", (900, 900))
    canvas = window.get_canvas()
    scene  = window.get_scene()

    camera = ti.ui.Camera()
    camera.position(0.0, 0.5, 2.0)
    camera.lookat(0.0, 0.0, 0.0)

    current_method = 1
    paused = False

    while window.running:

        # ── 控制面板 ──────────────────────────────────────────
        window.GUI.begin("Control Panel", 0.02, 0.02, 0.42, 0.52)

        window.GUI.text("=== Integration Method ===")
        prefix_0 = "[*] " if current_method == 0 else "[ ] "
        prefix_1 = "[*] " if current_method == 1 else "[ ] "
        prefix_2 = "[*] " if current_method == 2 else "[ ] "

        if window.GUI.button(prefix_0 + "Explicit Euler  (Unstable)"):
            current_method = 0
            init_cloth()
        if window.GUI.button(prefix_1 + "Semi-Implicit Euler (Stable)"):
            current_method = 1
            init_cloth()
        if window.GUI.button(prefix_2 + "Implicit Euler (Fixed-point)"):
            current_method = 2
            init_cloth()

        window.GUI.text("")
        pause_label = "Resume Simulation" if paused else "Pause  Simulation"
        if window.GUI.button(pause_label):
            paused = not paused
        if window.GUI.button("Reset Cloth"):
            init_cloth()
            paused = False

        window.GUI.text("")
        window.GUI.text("=== Spring Info ===")
        window.GUI.text(f"Total springs: {num_springs[None]}")
        window.GUI.text(f"k_structural : {k_s}")
        window.GUI.text(f"k_shear      : {k_shear}")
        window.GUI.text(f"k_bending    : {k_bend}")
        window.GUI.text("")
        window.GUI.text("Right-drag: rotate camera")

        window.GUI.end()
        # ─────────────────────────────────────────────────────

        if not paused:
            for _ in range(40):
                if current_method == 0:
                    step_explicit()
                elif current_method == 1:
                    step_semi_implicit()
                else:
                    step_implicit_iter()

        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        scene.particles(x, radius=0.015, color=(0.2, 0.6, 1.0))
        scene.lines(x, indices=spring_indices, width=1.5, color=(0.8, 0.8, 0.8))

        canvas.scene(scene)
        window.show()

if __name__ == '__main__':
    main()