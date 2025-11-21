import numpy as np

# Физические константы
G = 9.81
RHO = 1.225  # Плотность воздуха
OMEGA_EARTH = 7.2921e-5


class Missile:
    def __init__(self, x, y, z, mass, fuel_mass, burn_time, thrust, drag_coeff, area):
        self.pos = np.array([x, y, z], dtype=float)
        self.vel = np.array([0.0, 0.0, 0.0], dtype=float)
        # Для симуляции копии
        self.init_args = (x, y, z, mass, fuel_mass, burn_time, thrust, drag_coeff, area)

        self.mass_empty = mass
        self.fuel_mass = fuel_mass
        self.current_mass = mass + fuel_mass
        self.burn_time = burn_time
        self.thrust_force = thrust
        self.cd = drag_coeff
        self.area = area
        self.time_elapsed = 0.0

    def copy(self):
        """Создает точную копию ракеты для виртуальных расчетов"""
        return Missile(*self.init_args)

    def update(self, dt, wind_vector, launch_direction):
        """
        launch_direction: Нормализованный вектор (x, y, z), куда направлен нос ракеты.
        Если мы хотим, чтобы ракета сбивалась ветром, этот вектор должен быть зафиксирован при старте!
        """

        # 1. Сжигание топлива
        if self.time_elapsed < self.burn_time:
            dm = self.fuel_mass / self.burn_time
            self.current_mass -= dm * dt

        if self.current_mass < self.mass_empty:
            self.current_mass = self.mass_empty

        # 2. Силы

        # А. ТЯГА (Thrust)
        # Самый важный момент: Тяга направлена строго туда, куда мы ее направили при расчете.
        # Она НЕ корректируется в полете. Поэтому изменение ветра приведет к промаху.
        F_thrust = np.array([0.0, 0.0, 0.0])
        if self.time_elapsed < self.burn_time and launch_direction is not None:
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

        # Сумма сил
        F_total = F_thrust + F_gravity + F_drag

        # 3. Интеграция (Эйлер)
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


# --- АЛГОРИТМ РАСЧЕТА ТОЧНОГО ПОПАДАНИЯ ---
def find_perfect_trajectory(missile_template, target_config, initial_wind):
    """
    Запускает виртуальные симуляции, подбирая идеальный вектор запуска.
    Гарантирует попадание при initial_wind.
    """
    print(f"Calculations started. Wind: {initial_wind}")

    # 1. Данные цели
    t_pos_0 = np.array([target_config['x'], target_config['y'], target_config['z']])
    t_vel = np.array([target_config['vx'], target_config['vy'], target_config['vz']])

    # 2. Начальная догадка (Aim Point)
    # Предположим время полета 8 секунд (примерно)
    # Целимся в точку, где цель будет через 8 секунд + поправка вверх на гравитацию
    estimated_time = 8.0
    aim_point = t_pos_0 + t_vel * estimated_time
    aim_point[2] += 0.5 * G * (estimated_time ** 2)  # Компенсация падения

    # Нормализуем вектор направления
    launch_dir = aim_point / np.linalg.norm(aim_point)

    # 3. Цикл подбора (Zeroing)
    # Делаем до 15 попыток уточнить прицел
    best_dir = launch_dir
    min_error = 1e9

    for i in range(15):
        # Создаем виртуальную ракету и цель
        sim_missile = missile_template.copy()
        sim_t_pos = t_pos_0.copy()

        dt = 0.1  # Быстрый шаг для расчета
        sim_time = 0
        closest_dist = 1e9
        miss_vector = np.array([0, 0, 0])

        # Прогоняем полет (макс 15 сек)
        while sim_time < 15.0:
            m_p = np.array(sim_missile.update(dt, initial_wind, best_dir))
            t_p = sim_t_pos + t_vel * sim_time

            dist = np.linalg.norm(m_p - t_p)

            # Ищем точку максимального сближения
            if dist < closest_dist:
                closest_dist = dist
                miss_vector = t_p - m_p  # Вектор от ракеты к цели (ошибка)

            sim_time += dt

        # Если попали достаточно близко (< 1 метра) — стоп
        if closest_dist < 1.0:
            print(f"Solution found at iter {i}: Miss {closest_dist:.2f}m")
            return best_dir

        # КОРРЕКЦИЯ
        # Если мы промахнулись, нужно сместить точку прицеливания в сторону ошибки
        # Мы просто добавляем вектор ошибки к текущему направлению (с небольшим коэффициентом)
        correction_factor = 1.5  # Чем больше, тем агрессивнее поправка

        # Восстанавливаем "точку в пространстве", куда мы целились
        # (это условно, так как мы храним только direction, но для коррекции сойдет)
        aim_point = best_dir * 10000  # Условно на 10км
        aim_point += miss_vector * correction_factor

        best_dir = aim_point / np.linalg.norm(aim_point)

        if closest_dist < min_error:
            min_error = closest_dist

    print(f"Best solution found: Miss {min_error:.2f}m")
    return best_dir