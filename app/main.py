from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
import asyncio
import numpy as np
from .models import Missile, Target, find_perfect_trajectory

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

        # 3. Определяем РАКЕТУ
        missile_template = Missile(
            x=0, y=0, z=0,
            mass=60.0,
            fuel_mass=40.0,
            burn_time=10.0,
            thrust=8000.0,  # Достаточная тяга
            drag_coeff=0.2,
            area=0.05
        )

        # 4. ГЛАВНЫЙ РАСЧЕТ (Solving)
        # Рассчитываем, куда нужно повернуть пусковую установку, чтобы попасть
        # при ЭТОМ начальном ветре. Запуск расчёта в отдельном потоке, чтобы
        # не блокировать loop при тяжёлой виртуальной симуляции.
        perfect_launch_vector = await asyncio.to_thread(
            find_perfect_trajectory, missile_template, target_config, initial_wind
        )

        # 5. Запуск реальной симуляции
        missile = missile_template.copy()
        target = Target(**target_config)

        # Текущий ветер (может меняться пользователем)
        current_wind = initial_wind.copy()

        dt = 0.05
        max_time = 15.0
        current_time = 0.0

        while current_time < max_time:
            # Проверка обновлений от клиента (изменение ветра в реал-тайме)
            try:
                # Небольшой ненулевой таймаут, чтобы не перегружать loop
                incoming = await asyncio.wait_for(websocket.receive_json(), timeout=0.02)
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

            response = {
                "time": round(current_time, 2),
                "missile": m_pos,
                "target": t_pos,
                "distance": round(dist, 2),
                "wind": current_wind.tolist()
            }
            await websocket.send_json(response)

            current_time += dt
            await asyncio.sleep(dt)

        # Гарантированно закрываем соединение по завершении цикла
        try:
            await websocket.close()
        except Exception:
            logger.exception("Error while closing websocket")
            pass

    except WebSocketDisconnect:
        logger.info("Client disconnected (outer)")
    except Exception as e:
        logger.exception("Error in websocket handler")