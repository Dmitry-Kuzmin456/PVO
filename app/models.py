import numpy as np
from loguru import logger

# Физические константы
G = 9.81
RHO = 1.225  # Плотность воздуха
OMEGA_EARTH = 7.2921e-5
NEW_YORK_LATITUDE = 40.7  # Широта Нью-Йорка


class Missile:
    def __init__(self, x, y, z, mass, fuel_mass, burn_time, thrust, drag_coeff, area, latitude=NEW_YORK_LATITUDE):
        self.pos = np.array([x, y, z], dtype=float)
        self.vel = np.array([0.0, 0.0, 0.0], dtype=float)
        # Для симуляции копии
        self.latitude = latitude
        self.init_args = (x, y, z, mass, fuel_mass, burn_time, thrust, drag_coeff, area, latitude)

        self.mass_empty = mass
        self.fuel_mass = fuel_mass
        self.fuel_remaining = fuel_mass
        self.current_mass = mass + fuel_mass
        self.burn_time = burn_time
        self.thrust_force = thrust
        self.cd = drag_coeff
        self.area = area
        self.time_elapsed = 0.0

    def copy(self):
        """Создает точную копию ракеты для виртуальных расчетов"""
        return Missile(*self.init_args)

    def _calculate_coriolis_force(self, velocity):
        """Расчет силы Кориолиса для текущей широты"""
        omega_earth = 7.2921e-5  # рад/с

        # Преобразование широты в радианы
        lat_rad = np.radians(self.latitude)

        # Вектор угловой скорости Земли для данной широты
        # Для северного полушария: ω = [0, ω*cos(φ), ω*sin(φ)]
        omega_vector = np.array([
            0,
            omega_earth * np.cos(lat_rad),
            omega_earth * np.sin(lat_rad)
        ])

        # Сила Кориолиса: F = -2m(ω × v)
        coriolis_force = -2 * self.current_mass * np.cross(omega_vector, velocity)

        return coriolis_force

    def update(self, dt, wind_vector, launch_direction):
        """
        launch_direction: Нормализованный вектор (x, y, z), куда направлен нос ракеты.
        Если мы хотим, чтобы ракета сбивалась ветром, этот вектор должен быть зафиксирован при старте!
        """

        # 1. Сжигание топлива
        # Расход топлива: уменьшаем `fuel_remaining` и массу только пока есть топливо
        if self.time_elapsed < self.burn_time and self.fuel_remaining > 0.0:
            dm = self.fuel_mass / self.burn_time
            consumed = dm * dt
            if consumed > self.fuel_remaining:
                consumed = self.fuel_remaining
            self.fuel_remaining -= consumed
            self.current_mass -= consumed

        if self.current_mass < self.mass_empty:
            self.current_mass = self.mass_empty

        # 2. Силы

        # А. ТЯГА (Thrust)
        # Самый важный момент: Тяга направлена строго туда, куда мы ее направили при расчете.
        # Она НЕ корректируется в полете. Поэтому изменение ветра приведет к промаху.
        F_thrust = np.array([0.0, 0.0, 0.0])
        # Тяга действует только если ещё есть топливо и время горения
        if self.time_elapsed < self.burn_time and launch_direction is not None and self.fuel_remaining > 0.0:
            F_thrust = launch_direction * self.thrust_force

        # Б. ГРАВИТАЦИЯ
        F_gravity = np.array([0.0, 0.0, -self.current_mass * G])

        # В. АЭРОДИНАМИКА (Drag)
        # F_drag зависит от относительной скорости (V_missile - V_wind)
        # Именно здесь изменение ветра влияет на полет.
        v_rel = self.vel - np.array(wind_vector, dtype=float)
        v_rel_mag = np.linalg.norm(v_rel)

        F_drag = np.array([0.0, 0.0, 0.0])
        if v_rel_mag > 0:
            force_mag = 0.5 * RHO * self.cd * self.area * (v_rel_mag ** 2)
            F_drag = -force_mag * (v_rel / v_rel_mag)

        # Г. СИЛА КОРИОЛИСА (НОВОЕ!)
        F_coriolis = self._calculate_coriolis_force(self.vel)

        # Сумма сил
        F_total = F_thrust + F_gravity + F_drag + F_coriolis

        # 3. Интеграция — используем RK4 для лучшей точности
        # Состояние: pos (3), vel (3), fuel_remaining (1), time_elapsed (1)

        def derivatives(pos, vel, fuel_rem, time_el, wind_vec):
            # Масса в текущий момент
            current_mass = self.mass_empty + max(fuel_rem, 0.0)

            # Тяга действует только если есть топливо и время горения
            F_t = np.array([0.0, 0.0, 0.0])
            if time_el < self.burn_time and fuel_rem > 0 and launch_direction is not None:
                F_t = launch_direction * self.thrust_force

            F_g = np.array([0.0, 0.0, -current_mass * G])

            v_rel_local = vel - np.array(wind_vec, dtype=float)
            v_rel_mag_local = np.linalg.norm(v_rel_local)
            F_d = np.array([0.0, 0.0, 0.0])
            if v_rel_mag_local > 0:
                force_mag_local = 0.5 * RHO * self.cd * self.area * (v_rel_mag_local ** 2)
                F_d = -force_mag_local * (v_rel_local / v_rel_mag_local)

            # СИЛА КОРИОЛИСА в производных (НОВОЕ!)
            F_c = self._calculate_coriolis_force(vel)

            F_tot = F_t + F_g + F_d + F_c
            acc_local = F_tot / max(current_mass, 1e-6)

            # Расход топлива (dm/dt)
            fuel_rate = 0.0
            if time_el < self.burn_time and fuel_rem > 0:
                fuel_rate = -(self.fuel_mass / self.burn_time)

            # Возвращаем производные: pos' = vel, vel' = acc, fuel_rem' = fuel_rate, time' = 1
            return vel, acc_local, fuel_rate, 1.0

        # начальные значения
        p0 = self.pos.copy()
        v0 = self.vel.copy()
        f0 = self.fuel_remaining
        t0 = self.time_elapsed

        # RK4 шаг
        k1_v, k1_a, k1_f, k1_t = derivatives(p0, v0, f0, t0, wind_vector)
        p1 = p0 + (dt / 2.0) * k1_v
        v1 = v0 + (dt / 2.0) * k1_a
        f1 = f0 + (dt / 2.0) * k1_f
        t1 = t0 + (dt / 2.0) * k1_t

        k2_v, k2_a, k2_f, k2_t = derivatives(p1, v1, f1, t1, wind_vector)
        p2 = p0 + (dt / 2.0) * k2_v
        v2 = v0 + (dt / 2.0) * k2_a
        f2 = f0 + (dt / 2.0) * k2_f
        t2 = t0 + (dt / 2.0) * k2_t

        k3_v, k3_a, k3_f, k3_t = derivatives(p2, v2, f2, t2, wind_vector)
        p3 = p0 + dt * k3_v
        v3 = v0 + dt * k3_a
        f3 = f0 + dt * k3_f
        t3 = t0 + dt * k3_t

        k4_v, k4_a, k4_f, k4_t = derivatives(p3, v3, f3, t3, wind_vector)

        dp = (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
        dv = (dt / 6.0) * (k1_a + 2.0 * k2_a + 2.0 * k3_a + k4_a)
        df = (dt / 6.0) * (k1_f + 2.0 * k2_f + 2.0 * k3_f + k4_f)
        dt_time = dt  # интегрирование времени явно

        # Обновляем состояние
        self.pos = self.pos + dp
        self.vel = self.vel + dv
        self.fuel_remaining = max(self.fuel_remaining + df, 0.0)
        self.current_mass = self.mass_empty + self.fuel_remaining
        self.time_elapsed += dt_time

        # Защита на массу
        if self.current_mass < self.mass_empty:
            self.current_mass = self.mass_empty

        return self.pos.tolist()


class Target:
    def __init__(self, x, y, z, vx, vy, vz):
        self.pos = np.array([x, y, z], dtype=float)
        self.vel = np.array([vx, vy, vz], dtype=float)

    def update(self, dt):
        self.pos += self.vel * dt
        return self.pos.tolist()


# --- АЛГОРИТМ РАСЧЕТА ТОЧНОГО ПОПАДАНИЯ ---
def find_perfect_trajectory(missile_template, target_config, initial_wind,
                            sim_time_max=40.0,
                            az_steps=72, el_steps=36,
                            coarse_dt=0.01,
                            refine_dt=0.005,
                            final_dt=0.0025):
    """
    Запускает виртуальные симуляции, подбирая идеальный вектор запуска.
    Параметры позволяют управлять точностью и временем симуляции.
    """
    logger.info(
        f"Calculations started. Wind: {initial_wind} | sim_time_max={sim_time_max} az_steps={az_steps} el_steps={el_steps} coarse_dt={coarse_dt} refine_dt={refine_dt} final_dt={final_dt}")

    # 1. Данные цели
    t_pos_0 = np.array([target_config['x'], target_config['y'], target_config['z']])
    t_vel = np.array([target_config['vx'], target_config['vy'], target_config['vz']])

    # NEW ALGORITHM: grid search over azimuth/elevation with local refinement
    # This searches directions in spherical coords (azimuth, elevation) and
    # simulates the missile flight for each candidate direction, returning
    # the direction that minimizes the closest approach to the moving target.

    def simulate_closest_distance(direction_vec, sim_dt=coarse_dt, sim_time_max_local=sim_time_max):
        sim_missile = missile_template.copy()
        sim_t_pos = t_pos_0.copy()
        sim_time = 0.0
        closest = 1e9
        # Simulate until the time limit; use provided sim_dt
        while sim_time < sim_time_max_local:
            m_p = np.array(sim_missile.update(sim_dt, initial_wind, direction_vec))
            t_p = sim_t_pos + t_vel * sim_time
            d = np.linalg.norm(m_p - t_p)
            if d < closest:
                closest = d
            sim_time += sim_dt
        return closest

    # helper: convert azimuth [deg], elevation [deg] to unit vector
    def dir_from_angles(az_deg, el_deg):
        az = np.radians(az_deg)
        el = np.radians(el_deg)
        x = np.cos(el) * np.cos(az)
        y = np.cos(el) * np.sin(az)
        z = np.sin(el)
        v = np.array([x, y, z], dtype=float)
        n = np.linalg.norm(v)
        return v / n if n != 0 else np.array([1.0, 0.0, 0.0])

    # Coarse grid search (denser and finer dt for better coverage)
    best_dir = None
    best_dist = 1e9

    # search ranges
    az_range = (0, 360)
    el_range = (-20, 85)  # degrees -- expanded elevation range

    for az in np.linspace(az_range[0], az_range[1], az_steps, endpoint=False):
        for el in np.linspace(el_range[0], el_range[1], el_steps):
            vec = dir_from_angles(az, el)
            d = simulate_closest_distance(vec, sim_dt=coarse_dt, sim_time_max_local=sim_time_max)
            if d < best_dist:
                best_dist = d
                best_dir = vec

    # local refinements around best_dir in angle space
    if best_dir is None:
        return np.array([1.0, 0.0, 0.0])

    # get az/el of best_dir
    best_az = np.degrees(np.arctan2(best_dir[1], best_dir[0])) % 360
    best_el = np.degrees(np.arcsin(best_dir[2]))

    # Local refinements around best_dir in angle space (multi-stage, decreasing spans)
    refine_steps = [10, 8, 6]
    refine_span = [12.0, 3.0, 0.8]  # degrees
    for steps, span in zip(refine_steps, refine_span):
        az_vals = np.linspace(best_az - span, best_az + span, steps)
        el_vals = np.linspace(best_el - span, best_el + span, steps)
        for az in az_vals:
            for el in el_vals:
                vec = dir_from_angles(az, el)
                d = simulate_closest_distance(vec, sim_dt=refine_dt, sim_time_max_local=sim_time_max)
                if d < best_dist:
                    best_dist = d
                    best_dir = vec
                    best_az = az
                    best_el = el

    # Further local optimization: coordinate descent on az/el with decreasing step sizes
    step_sizes = [0.5, 0.2, 0.1, 0.05]
    for step in step_sizes:
        improved = True
        while improved:
            improved = False
            for d_az, d_el in [(step, 0), (-step, 0), (0, step), (0, -step)]:
                cand_az = best_az + d_az
                cand_el = best_el + d_el
                vec = dir_from_angles(cand_az, cand_el)
                d = simulate_closest_distance(vec, sim_dt=final_dt, sim_time_max_local=sim_time_max)
                if d + 1e-6 < best_dist:
                    best_dist = d
                    best_dir = vec
                    best_az = cand_az
                    best_el = cand_el
                    improved = True

    # Final refinement using a small Nelder-Mead optimizer in az/el space
    def nelder_mead_2d(obj_fn, x0, step0=0.5, tol=1e-3, max_iter=200, target=3.0):
        # x in degrees [az, el]
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5

        # initial simplex: x0, x0 + [step,0], x0 + [0,step]
        simplex = [np.array(x0, dtype=float),
                   np.array([x0[0] + step0, x0[1]], dtype=float),
                   np.array([x0[0], x0[1] + step0], dtype=float)]
        fvals = [obj_fn(p) for p in simplex]
        it = 0
        while it < max_iter:
            # order
            idx = np.argsort(fvals)
            simplex = [simplex[i] for i in idx]
            fvals = [fvals[i] for i in idx]
            best = simplex[0]
            worst = simplex[-1]
            second = simplex[1]

            # termination
            if np.std(fvals) < tol or fvals[0] <= target:
                break

            # centroid of best and second
            centroid = (simplex[0] + simplex[1]) / 2.0

            # reflection
            xr = centroid + alpha * (centroid - worst)
            fr = obj_fn(xr)
            if fvals[0] <= fr < fvals[1]:
                simplex[-1] = xr
                fvals[-1] = fr
            elif fr < fvals[0]:
                # expansion
                xe = centroid + gamma * (xr - centroid)
                fe = obj_fn(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            else:
                # contraction
                xc = centroid + rho * (worst - centroid)
                fc = obj_fn(xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = obj_fn(simplex[i])
            it += 1

        return simplex[0], fvals[0], it

    # objective for Nelder-Mead: given [az,el] degrees, return closest distance
    def nm_obj(x_deg):
        az, el = float(x_deg[0]), float(x_deg[1])
        vec = dir_from_angles(az, el)
        return simulate_closest_distance(vec, sim_dt=final_dt, sim_time_max_local=sim_time_max)

    try:
        start = np.array([best_az, best_el], dtype=float)
        nm_start, nm_val, nm_iters = nelder_mead_2d(nm_obj, start, step0=0.5, tol=1e-3, max_iter=300, target=3.0)
        if nm_val + 1e-9 < best_dist:
            best_dist = nm_val
            best_dir = dir_from_angles(float(nm_start[0]), float(nm_start[1]))
            best_az = float(nm_start[0])
            best_el = float(nm_start[1])
        logger.info(f"Nelder-Mead finished: val={nm_val:.3f} iters={nm_iters} best_dist={best_dist:.3f}")
    except Exception:
        logger.exception("Nelder-Mead refinement failed; keeping previous best")

    logger.info(f"Best solution found: Miss {best_dist:.2f}m")
    return best_dir