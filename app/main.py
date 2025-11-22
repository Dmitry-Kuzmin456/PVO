from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
import asyncio
import numpy as np
from .models import Missile, Target, find_perfect_trajectory, NEW_YORK_LATITUDE

# Simple in-memory cache for computed launch vectors
_LAUNCH_VECTOR_CACHE = {}

def _make_cache_key(missile_template, target_config, wind_vec, precision):
    # Build a simple tuple key based on rounded parameters
    m = missile_template
    key = (
        round(wind_vec[0], 3), round(wind_vec[1], 3), round(wind_vec[2], 3),
        int(target_config['x']), int(target_config['y']), int(target_config['z']),
        int(target_config['vx']), int(target_config['vy']), int(target_config['vz']),
        round(m.mass_empty, 3), round(m.fuel_mass, 3), round(m.thrust_force, 3),
        precision,
        round(m.latitude, 1)  # Добавляем широту в ключ кэша
    )
    return key

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/ws/simulate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        # 1. Получаем начальные настройки от пользователя
        data = await websocket.receive_json()
        try:
            start_ws = float(data.get('wind_speed'))
            start_wd = float(data.get('wind_dir'))
        except (TypeError, ValueError):
            await websocket.send_json({"error": "invalid wind_speed or wind_dir"})
            await websocket.close()
            return

        # Небольшая задержка для демонстрации индикатора загрузки (опционально)
        # await asyncio.sleep(1)

        # Формируем вектор начального ветра
        wx = start_ws * np.cos(np.radians(start_wd))
        wy = start_ws * np.sin(np.radians(start_wd))
        initial_wind = np.array([wx, wy, 0.0], dtype=float)

        # 2. Определяем ЦЕЛЬ (прямо в коде, как просили)
        # Цель летит на встречу и немного вниз
        target_config = {
            'x': 4000, 'y': 6000, 'z': 4000,  # Старт далеко
            'vx': -50, 'vy': -300, 'vz': -50  # Скорость (м/с)
        }

        # 3. Определяем РАКЕТУ (с широтой Нью-Йорка)
        missile_template = Missile(
            x=0, y=0, z=0,
            mass=60.0,
            fuel_mass=40.0,
            burn_time=10.0,
            thrust=8000.0,  # Достаточная тяга
            drag_coeff=0.2,
            area=0.05,
            latitude=NEW_YORK_LATITUDE  # Широта Нью-Йорка
        )

        # 4. ГЛАВНЫЙ РАСЧЕТ (Solving)
        # Рассчитываем, куда нужно повернуть пусковую установку, чтобы попасть
        # при ЭТОМ начальном ветре. Запуск расчёта в отдельном потоке, чтобы
        # не блокировать loop при тяжёлой виртуальной симуляции.
        precision = data.get('precision', 'fast')

        cache_key = _make_cache_key(missile_template, target_config, initial_wind.tolist(), precision)
        perfect_launch_vector = None
        if cache_key in _LAUNCH_VECTOR_CACHE:
            perfect_launch_vector = _LAUNCH_VECTOR_CACHE[cache_key]
            logger.info("Using cached launch vector")
        else:
            # choose parameters for fast vs accurate
            if precision == 'fast':
                # fast: coarser grid, shorter sim time, larger dt
                params = dict(sim_time_max=20.0, az_steps=36, el_steps=18,
                              coarse_dt=0.02, refine_dt=0.01, final_dt=0.005)
            else:
                # accurate: defaults in function (higher fidelity)
                params = {}

            logger.info(f"Calculating new trajectory with precision: {precision}")
            perfect_launch_vector = await asyncio.to_thread(
                find_perfect_trajectory, missile_template, target_config, initial_wind, **params
            )
            # cache result
            try:
                _LAUNCH_VECTOR_CACHE[cache_key] = perfect_launch_vector
                logger.info("Cached new launch vector")
            except Exception as e:
                logger.error(f"Failed to cache result: {e}")

        # 5. Запуск реальной симуляции
        missile = missile_template.copy()
        target = Target(**target_config)

        # Текущий ветер (может меняться пользователем)
        current_wind = initial_wind.copy()

        dt = 0.01
        max_time = 25.0
        current_time = 0.0
        collision_threshold = 3.0  # meters — target threshold for hit (user-requested)
        prev_dist = float('inf')
        last_m_pos = None
        last_t_pos = None
        # Отправляем данные клиенту с интервалом 0.01 с (повышенная частота)
        send_interval = 0.01  # seconds between frames sent to client
        last_send_time = -send_interval
        # Для надёжной детекции пересечения: отслеживаем минимум дистанции и позицию в этот момент
        min_dist = float('inf')
        min_m_pos = None
        min_t_pos = None
        min_time = None
        min_m_vel = None
        got_hit = False

        while current_time < max_time:
            # Проверка обновлений от клиента (изменение ветра в реал-тайме)
            try:
                # Небольшой ненулевой таймаут, чтобы не перегружать loop
                incoming = await asyncio.wait_for(websocket.receive_json(), timeout=0.005)
                if 'wind_speed' in incoming:
                    ws = float(incoming['wind_speed'])
                    wd = float(incoming['wind_dir'])
                    # Обновляем текущий ветер
                    current_wind[0] = ws * np.cos(np.radians(wd))
                    current_wind[1] = ws * np.sin(np.radians(wd))
            except asyncio.TimeoutError:
                pass  # Ничего не пришло, продолжаем с текущим ветром
            except asyncio.CancelledError:
                break
            except WebSocketDisconnect:
                logger.info("Client disconnected during receive")
                break
            except Exception as e:
                logger.exception("Unexpected error receiving websocket message")
                break

            # ФИЗИКА
            # Передаем perfect_launch_vector. Ракета пытается лететь по нему.
            # Если current_wind отличается от initial_wind, аэродинамика сдвинет ракету -> промах.
            m_pos = missile.update(dt, current_wind, launch_direction=perfect_launch_vector)
            t_pos = target.update(dt)

            dist = np.linalg.norm(np.array(m_pos) - np.array(t_pos))

            # Обновляем минимум дистанции и запоминаем позиции в момент минимума
            if dist < min_dist:
                min_dist = dist
                min_m_pos = m_pos
                min_t_pos = t_pos
                min_time = current_time
                try:
                    min_m_vel = np.array(missile.vel, dtype=float)
                except Exception:
                    min_m_vel = None

            # Если дистанция упала ниже порога — считаем попадание
            if dist <= collision_threshold:
                missile.pos = np.array(t_pos, dtype=float)
                try:
                    missile.vel = np.array([0.0, 0.0, 0.0], dtype=float)
                except Exception:
                    pass

                response = {
                    "time": f"{current_time:.2f}",
                    "missile": missile.pos.tolist(),
                    "missile_velocity": missile.vel.tolist(),
                    "missile_speed": round(float(np.linalg.norm(missile.vel)), 2),
                    "target": t_pos,
                    "distance": round(dist, 2),
                    "wind": current_wind.tolist(),
                    "hit": True
                }

                await websocket.send_json(response)
                got_hit = True
                break

            # Обнаружение прохождения цели между шагами: если дистанция начала расти после минимума
            if dist > prev_dist and min_dist <= (collision_threshold * 1.2):
                # Используем позицию в момент минимума как точку столкновения
                collision_point = min_t_pos if min_t_pos is not None else t_pos
                missile.pos = np.array(collision_point, dtype=float)
                try:
                    missile.vel = np.array([0.0, 0.0, 0.0], dtype=float)
                except Exception:
                    pass

                response = {
                    "time": f"{current_time:.2f}",
                    "missile": missile.pos.tolist(),
                    "missile_velocity": missile.vel.tolist(),
                    "missile_speed": round(float(np.linalg.norm(missile.vel)), 2),
                    "target": collision_point,
                    "distance": round(min_dist, 2),
                    "wind": current_wind.tolist(),
                    "hit": True,
                    "note": "passed-through-detected"
                }

                await websocket.send_json(response)
                got_hit = True
                break

            # Отправляем фреймы клиенту с шагом send_interval
            if (current_time - last_send_time) >= send_interval:
                response = {
                    "time": f"{current_time:.2f}",
                    "missile": m_pos,
                    "missile_velocity": missile.vel.tolist(),
                    "missile_speed": round(float(np.linalg.norm(missile.vel)), 2),
                    "target": t_pos,
                    "distance": round(dist, 2),
                    "wind": current_wind.tolist(),
                    "hit": False
                }
                await websocket.send_json(response)
                last_send_time = current_time

            # Сохраняем предыдущие значения для детекции прохождения
            prev_dist = dist
            last_m_pos = m_pos
            last_t_pos = t_pos

            current_time += dt
            await asyncio.sleep(dt)

        # Гарантированно закрываем соединение по завершении цикла
        # Если цикл завершился без попадания — отправляем итоговую сводку
        if not got_hit:
            try:
                final_summary = {
                    "time": f"{current_time:.2f}",
                    "missile": missile.pos.tolist(),
                    "missile_velocity": missile.vel.tolist(),
                    "missile_speed": round(float(np.linalg.norm(missile.vel)), 2),
                    "target": t_pos,
                    "distance": round(dist, 2),
                    "min_distance": (round(min_dist, 2) if min_dist != float('inf') else None),
                    "closest_missile": (min_m_pos if min_m_pos is not None else None),
                    "closest_target": (min_t_pos if min_t_pos is not None else None),
                    "closest_time": (f"{min_time:.2f}" if min_time is not None else None),
                    "closest_missile_speed": (round(float(np.linalg.norm(min_m_vel)), 2) if min_m_vel is not None else None),
                    "wind": current_wind.tolist(),
                    "hit": False,
                    "note": "simulation_finished_no_hit"
                }
                await websocket.send_json(final_summary)
            except Exception:
                logger.exception("Failed to send final summary to client")
        try:
            await websocket.close()
        except Exception:
            logger.exception("Error while closing websocket")
            pass

    except WebSocketDisconnect:
        logger.info("Client disconnected (outer)")
    except Exception as e:
        logger.exception("Error in websocket handler")