import numpy as np

# Константы
G = 9.81  # Ускорение свободного падения (м/с^2)
RHO = 1.225  # Плотность воздуха (кг/м^3)
OMEGA_EARTH = 7.2921e-5  # Угловая скорость вращения Земли (рад/с)
LATITUDE = np.radians(45)  # Широта для расчета Кориолиса (для примера 45 град)


class Missile:
    def __init__(self, x, y, z, mass, fuel_mass, burn_time, thrust, drag_coeff, area):
        self.pos = np.array([x, y, z], dtype=float)
        self.vel = np.array([0.0, 0.0, 0.0], dtype=float)
        self.mass_empty = mass
        self.fuel_mass = fuel_mass
        self.current_mass = mass + fuel_mass
        self.burn_time = burn_time
        self.thrust_force = thrust
        self.cd = drag_coeff
        self.area = area
        self.time_elapsed = 0.0

    def get_mass_derivative(self):
        if self.time_elapsed < self.burn_time:
            return -self.fuel_mass / self.burn_time
        return 0.0

    def update(self, dt, wind_vector, launch_direction=None):
        # 1. Масса
        dm = self.get_mass_derivative()
        if self.current_mass > self.mass_empty:
            self.current_mass += dm * dt

        # 2. Силы
        # А. Тяга (направлена по вектору скорости или начальному вектору запуска)
        # Для простоты: если скорость > 0, тяга по скорости, иначе по запуску
        if np.linalg.norm(self.vel) > 0.1:
            v_norm = self.vel / np.linalg.norm(self.vel)
        else:
            v_norm = launch_direction if launch_direction is not None else np.array([0, 0, 1])

        F_thrust = np.array([0.0, 0.0, 0.0])
        if self.time_elapsed < self.burn_time:
            F_thrust = v_norm * self.thrust_force

        # Б. Гравитация
        F_gravity = np.array([0.0, 0.0, -self.current_mass * G])

        # В. Аэродинамическое сопротивление
        # V_rel = V_missile - V_wind
        v_rel = self.vel - wind_vector
        v_rel_mag = np.linalg.norm(v_rel)

        F_drag = np.array([0.0, 0.0, 0.0])
        if v_rel_mag > 0:
            force_mag = 0.5 * RHO * self.cd * self.area * (v_rel_mag ** 2)
            F_drag = -force_mag * (v_rel / v_rel_mag)

        # Г. Сила Кориолиса
        # Вектор вращения Земли (в локальной системе: Север, Восток, Верх -> Y, X, Z)
        # Упрощенно: Omega = [0, Omega * cos(lat), Omega * sin(lat)]
        Omega_vec = np.array([0, OMEGA_EARTH * np.cos(LATITUDE), OMEGA_EARTH * np.sin(LATITUDE)])
        F_coriolis = -2 * self.current_mass * np.cross(Omega_vec, self.vel)

        # Сумма сил
        F_total = F_thrust + F_gravity + F_drag + F_coriolis

        # Интегрирование (Метод Эйлера для скорости real-time)
        acc = F_total / self.current_mass
        self.vel += acc * dt
        self.pos += self.vel * dt

        self.time_elapsed += dt

        return self.pos.tolist()


class Target:
    def __init__(self, x, y, z, vx, vy, vz):
        self.pos = np.array([x, y, z], dtype=float)
        self.vel = np.array([vx, vy, vz], dtype=float)

    def update(self, dt):
        self.pos += self.vel * dt
        return self.pos.tolist()


# Функция "Пристрелки" (Расчет траектории для попадания при начальных условиях)
def calculate_firing_solution(missile_params, target_params, initial_wind):
    # Это упрощенный алгоритм: мы стреляем "в упреждение"
    # Рассчитываем, где будет цель через примерно 7 секунд (половина времени симуляции)
    # и направляем вектор тяги туда.

    # В реальной системе тут была бы итеративная система решения краевой задачи.
    t_intercept = 7.0
    future_target_pos = np.array([
        target_params['x'] + target_params['vx'] * t_intercept,
        target_params['y'] + target_params['vy'] * t_intercept,
        target_params['z'] + target_params['vz'] * t_intercept
    ])

    start_pos = np.array([0, 0, 0])  # Ракета стартует из нуля

    direction = future_target_pos - start_pos
    direction = direction / np.linalg.norm(direction)  # Нормализация

    return direction