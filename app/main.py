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
_CALCULATION_RESULTS = {}  # Хранилище результатов расчетов


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


@app.websocket("/ws/calculate")
async def websocket_calculate(websocket: WebSocket):
    await websocket.accept()

    try:
        # Получаем данные для расчета
        data = await websocket.receive_json()

        # Проверяем, что это запрос на расчет
        if data.get('action') != 'calculate':
            await websocket.send_json({"error": "Invalid action"})
            await websocket.close()
            return

        try:
            start_ws = float(data.get('wind_speed'))
            start_wd = float(data.get('wind_dir'))
        except (TypeError, ValueError):
            await websocket.send_json({"error": "invalid wind_speed or wind_dir"})
            await websocket.close()
            return

        # Сообщаем клиенту о начале расчета
        await websocket.send_json({"status": "calculating"})

        # Формируем вектор начального ветра
        wx = start_ws * np.cos(np.radians(start_wd))
        wy = start_ws * np.sin(np.radians(start_wd))
        initial_wind = np.array([wx, wy, 0.0], dtype=float)

        # Определяем цель
        target_config = {
            'x': 4000, 'y': 6000, 'z': 4000,
            'vx': -50, 'vy': -300, 'vz': -50
        }

        # Определяем ракету
        missile_template = Missile(
            x=0, y=0, z=0,
            mass=60.0,
            fuel_mass=40.0,
            burn_time=10.0,
            thrust=8000.0,
            drag_coeff=0.2,
            area=0.05,
            latitude=NEW_YORK_LATITUDE
        )

        # Расчет оптимальной траектории
        precision = data.get('precision', 'fast')

        cache_key = _make_cache_key(missile_template, target_config, initial_wind.tolist(), precision)
        perfect_launch_vector = None

        if cache_key in _LAUNCH_VECTOR_CACHE:
            perfect_launch_vector = _LAUNCH_VECTOR_CACHE[cache_key]
            logger.info("Using cached launch vector")
        else:
            # Выбираем параметры в зависимости от точности
            if precision == 'fast':
                params = dict(sim_time_max=20.0, az_steps=36, el_steps=18,
                              coarse_dt=0.02, refine_dt=0.01, final_dt=0.005)
            else:
                params = {}

            logger.info(f"Calculating new trajectory with precision: {precision}")
            perfect_launch_vector = await asyncio.to_thread(
                find_perfect_trajectory, missile_template, target_config, initial_wind, **params
            )
            # Кэшируем результат
            try:
                _LAUNCH_VECTOR_CACHE[cache_key] = perfect_launch_vector
                logger.info("Cached new launch vector")
            except Exception as e:
                logger.error(f"Failed to cache result: {e}")

        # Сохраняем результат расчета
        calculation_id = str(hash(cache_key))
        _CALCULATION_RESULTS[calculation_id] = {
            'launch_vector': perfect_launch_vector.tolist(),
            'target_config': target_config,
            'missile_template': {
                'x': 0, 'y': 0, 'z': 0,
                'mass': 60.0, 'fuel_mass': 40.0, 'burn_time': 10.0,
                'thrust': 8000.0, 'drag_coeff': 0.2, 'area': 0.05,
                'latitude': NEW_YORK_LATITUDE
            }
        }

        # Отправляем результат клиенту
        await websocket.send_json({
            "status": "calculated",
            "calculation_id": calculation_id,
            "trajectory_data": {
                "launch_vector": perfect_launch_vector.tolist(),
                "cache_used": cache_key in _LAUNCH_VECTOR_CACHE
            }
        })

        await websocket.close()

    except WebSocketDisconnect:
        logger.info("Client disconnected from calculation")
    except Exception as e:
        logger.exception("Error in calculation websocket handler")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass


@app.websocket("/ws/simulate")
async def websocket_simulate(websocket: WebSocket):
    await websocket.accept()

    try:
        # Получаем данные для симуляции
        data = await websocket.receive_json()

        # Проверяем, что это запрос на симуляцию
        if data.get('action') != 'simulate':
            await websocket.send_json({"error": "Invalid action"})
            await websocket.close()
            return

        trajectory_data = data.get('trajectory_data')
        if not trajectory_data or 'launch_vector' not in trajectory_data:
            await websocket.send_json({"error": "No trajectory data provided"})
            await websocket.close()
            return

        try:
            start_ws = float(data.get('wind_speed'))
            start_wd = float(data.get('wind_dir'))
            launch_vector = np.array(trajectory_data['launch_vector'], dtype=float)
        except (TypeError, ValueError) as e:
            await websocket.send_json({"error": f"invalid parameters: {str(e)}"})
            await websocket.close()
            return

        # Формируем вектор начального ветра
        wx = start_ws * np.cos(np.radians(start_wd))
        wy = start_ws * np.sin(np.radians(start_wd))
        initial_wind = np.array([wx, wy, 0.0], dtype=float)

        # Определяем цель (такая же как при расчете)
        target_config = {
            'x': 4000, 'y': 6000, 'z': 4000,
            'vx': -50, 'vy': -300, 'vz': -50
        }

        # Создаем ракету и цель
        missile = Missile(
            x=0, y=0, z=0,
            mass=60.0,
            fuel_mass=40.0,
            burn_time=10.0,
            thrust=8000.0,
            drag_coeff=0.2,
            area=0.05,
            latitude=NEW_YORK_LATITUDE
        )
        target = Target(**target_config)

        # Текущий ветер (может меняться пользователем)
        current_wind = initial_wind.copy()

        # Параметры симуляции
        dt = 0.01
        max_time = 25.0
        current_time = 0.0
        collision_threshold = 3.0
        prev_dist = float('inf')
        send_interval = 0.01
        last_send_time = -send_interval
        min_dist = float('inf')
        min_m_pos = None
        min_t_pos = None
        min_time = None
        min_m_vel = None
        got_hit = False

        while current_time < max_time:
            # Проверка обновлений от клиента (изменение ветра в реал-тайме)
            try:
                incoming = await asyncio.wait_for(websocket.receive_json(), timeout=0.005)
                if 'wind_speed' in incoming:
                    ws = float(incoming['wind_speed'])
                    wd = float(incoming['wind_dir'])
                    current_wind[0] = ws * np.cos(np.radians(wd))
                    current_wind[1] = ws * np.sin(np.radians(wd))
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                break
            except WebSocketDisconnect:
                logger.info("Client disconnected during receive")
                break
            except Exception as e:
                logger.exception("Unexpected error receiving websocket message")
                break

            # ФИЗИКА - используем рассчитанный вектор запуска
            m_pos = missile.update(dt, current_wind, launch_direction=launch_vector)
            t_pos = target.update(dt)
            dist = np.linalg.norm(np.array(m_pos) - np.array(t_pos))

            # Обновляем минимум дистанции
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

            # Обнаружение прохождения цели между шагами
            if dist > prev_dist and min_dist <= (collision_threshold * 1.2):
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

            prev_dist = dist
            current_time += dt
            await asyncio.sleep(dt)

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
                    "closest_missile_speed": (
                        round(float(np.linalg.norm(min_m_vel)), 2) if min_m_vel is not None else None),
                    "wind": current_wind.tolist(),
                    "hit": False,
                    "note": "simulation_finished_no_hit"
                }
                await websocket.send_json(final_summary)
            except Exception:
                logger.exception("Failed to send final summary to client")

        await websocket.close()

    except WebSocketDisconnect:
        logger.info("Client disconnected from simulation")
    except Exception as e:
        logger.exception("Error in simulation websocket handler")