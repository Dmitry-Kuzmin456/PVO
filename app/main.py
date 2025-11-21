from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import asyncio
import json
import numpy as np
from .models import Missile, Target, calculate_firing_solution

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/ws/simulate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        # 1. Получаем начальные параметры от клиента
        data = await websocket.receive_json()

        # Начальный ветер (из настроек)
        wind_speed = float(data['wind_speed'])
        wind_dir = float(data['wind_dir'])  # в градусах

        # Конвертируем ветер в вектор (X, Y, 0)
        wx = wind_speed * np.cos(np.radians(wind_dir))
        wy = wind_speed * np.sin(np.radians(wind_dir))
        current_wind = np.array([wx, wy, 0.0])

        # Параметры цели (летит на высоте 5км)
        target_config = {
            'x': 2000, 'y': 5000, 'z': 3000,
            'vx': 100, 'vy': -200, 'vz': 0
        }

        # Параметры ракеты
        missile_config = {
            'x': 0, 'y': 0, 'z': 0,
            'mass': 50.0,  # пустая масса
            'fuel_mass': 40.0,  # топливо
            'burn_time': 10.0,  # время работы двигателя
            'thrust': 3000.0,  # сила тяги (Н)
            'drag_coeff': 0.3,
            'area': 0.05
        }

        # 2. Рассчитываем вектор запуска ИДЕАЛЬНО под начальный ветер
        # Ракета "программируется" на этот полет
        launch_vec = calculate_firing_solution(missile_config, target_config, current_wind)

        # Инициализация объектов
        missile = Missile(**missile_config)
        target = Target(**target_config)

        dt = 0.05  # Шаг времени 50мс
        max_time = 15.0
        current_time = 0.0

        # Цикл симуляции
        while current_time < max_time:
            # Проверяем, не прислал ли пользователь новые данные о ветре
            try:
                # Используем asyncio.wait_for с малым таймаутом для неблокирующего чтения
                incoming = await asyncio.wait_for(websocket.receive_json(), timeout=0.001)
                if 'wind_speed' in incoming:
                    ws = float(incoming['wind_speed'])
                    wd = float(incoming['wind_dir'])
                    current_wind[0] = ws * np.cos(np.radians(wd))
                    current_wind[1] = ws * np.sin(np.radians(wd))
            except asyncio.TimeoutError:
                pass  # Данных нет, продолжаем
            except Exception as e:
                break

            # Обновление физики
            m_pos = missile.update(dt, current_wind, launch_direction=launch_vec)
            t_pos = target.update(dt)

            # Расчет дистанции (промаха/попадания)
            distance = np.linalg.norm(np.array(m_pos) - np.array(t_pos))

            # Отправка данных
            response = {
                "time": round(current_time, 2),
                "missile": m_pos,
                "target": t_pos,
                "distance": round(distance, 2),
                "wind": current_wind.tolist()
            }
            await websocket.send_json(response)

            current_time += dt
            await asyncio.sleep(dt)  # Реальное время

        await websocket.close()

    except WebSocketDisconnect:
        print("Client disconnected")