let socket;
let isRunning = false;
let isCalculating = false;
let calculatedTrajectory = null;

const layout = {
    title: 'Траектория полета',
    scene: {
        aspectmode: 'data'
    },
    margin: {l: 0, r: 0, b: 0, t: 30}
};

// Инициализируем пустой график
Plotly.newPlot('graph', [{
    type: 'scatter3d', mode: 'lines+markers', name: 'Ракета',
    x: [0], y: [0], z: [0], marker: {color:'red', size:4}
}, {
    type: 'scatter3d', mode: 'lines+markers', name: 'Цель',
    x: [4000], y: [6000], z: [4000], marker: {color:'blue', size:4}
}], layout);

const wsInput = document.getElementById('wind-speed');
const wdInput = document.getElementById('wind-dir');

function updateInputs() {
    document.getElementById('ws-val').innerText = wsInput.value;
    document.getElementById('wd-val').innerText = wdInput.value;
    if (isRunning && socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({
            wind_speed: Number(wsInput.value),
            wind_dir: Number(wdInput.value)
        }));
    }
}

wsInput.oninput = updateInputs;
wdInput.oninput = updateInputs;

function showLoading() {
    document.getElementById('loading-indicator').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loading-indicator').style.display = 'none';
}

function showSimulateButton() {
    document.getElementById('simulate-btn').style.display = 'block';
}

function hideSimulateButton() {
    document.getElementById('simulate-btn').style.display = 'none';
}

function calculateTrajectory() {
    if (isCalculating) return;

    document.getElementById('calculate-btn').disabled = true;
    isCalculating = true;

    // Сбрасываем предыдущие результаты
    calculatedTrajectory = null;
    hideSimulateButton();

    // Показываем индикатор загрузки
    showLoading();

    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    socket = new WebSocket(`${protocol}://${window.location.host}/ws/calculate`);

    socket.onopen = function() {
        socket.send(JSON.stringify({
            wind_speed: Number(wsInput.value),
            wind_dir: Number(wdInput.value),
            precision: document.getElementById('precision-select').value,
            action: 'calculate'  // Указываем, что это расчет траектории
        }));
    };

    socket.onmessage = function(event) {
        const data = JSON.parse(event.data);

        if (data.status === 'calculating') {
            // Сервер сообщает, что начал расчет
            console.log('Расчет траектории начат...');
        }
        else if (data.status === 'calculated') {
            // Расчет завершен успешно
            calculatedTrajectory = data.trajectory_data;
            hideLoading();
            document.getElementById('calculate-btn').disabled = false;
            isCalculating = false;
            showSimulateButton();
            console.log('Траектория рассчитана успешно');
        }
        else if (data.error) {
            // Ошибка расчета
            hideLoading();
            document.getElementById('calculate-btn').disabled = false;
            isCalculating = false;
            console.error('Ошибка расчета:', data.error);
            alert('Ошибка при расчете траектории: ' + data.error);
        }
    };

    socket.onclose = function() {
        isCalculating = false;
        document.getElementById('calculate-btn').disabled = false;
        hideLoading();
        console.log("Calculation connection closed");
    };

    socket.onerror = function(error) {
        isCalculating = false;
        document.getElementById('calculate-btn').disabled = false;
        hideLoading();
        console.log("Calculation WebSocket error:", error);
    };
}

function startSimulation() {
    if (isRunning || !calculatedTrajectory) return;

    document.getElementById('simulate-btn').disabled = true;
    document.getElementById('calculate-btn').disabled = true;
    isRunning = true;

    // Очищаем графики
    Plotly.restyle('graph', {x: [[0]], y: [[0]], z: [[0]]}, [0]);
    Plotly.restyle('graph', {x: [[4000]], y: [[6000]], z: [[4000]]}, [1]);

    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    socket = new WebSocket(`${protocol}://${window.location.host}/ws/simulate`);

    // Буферы для накопления точек
    let mX = [0], mY = [0], mZ = [0];
    let tX = [4000], tY = [6000], tZ = [4000];

    socket.onopen = function() {
        socket.send(JSON.stringify({
            wind_speed: Number(wsInput.value),
            wind_dir: Number(wdInput.value),
            precision: document.getElementById('precision-select').value,
            action: 'simulate',  // Указываем, что это симуляция
            trajectory_data: calculatedTrajectory  // Передаем рассчитанную траекторию
        }));

        document.getElementById('min-dist-row').style.display = 'none';
        document.getElementById('closest-time-row').style.display = 'none';
    };

    socket.onmessage = function(event) {
        const data = JSON.parse(event.data);

        // time may be sent as formatted string from server; ensure two decimals
        const timeStr = (typeof data.time === 'string') ? data.time : Number(data.time).toFixed(2);
        document.getElementById('time-disp').innerText = timeStr;
        document.getElementById('alt-disp').innerText = Math.round(data.missile[2]);
        // format distance to two decimals
        document.getElementById('dist-disp').innerText = (typeof data.distance === 'number') ? data.distance.toFixed(2) : data.distance;
        // Update min distance and closest time if provided (final summary)
        if (data.min_distance !== undefined && data.min_distance !== null) {
            document.getElementById('min-dist-disp').innerText = Number(data.min_distance).toFixed(2);
            document.getElementById('min-dist-row').style.display = '';
        }
        if (data.closest_time !== undefined && data.closest_time !== null) {
            document.getElementById('closest-time-disp').innerText = data.closest_time;
            document.getElementById('closest-time-row').style.display = '';
        }
        // display missile speed if provided (during sim/hit it's current speed)
        if (data.missile_speed !== undefined) {
            document.getElementById('speed-disp').innerText = Number(data.missile_speed).toFixed(2);
        }

        // If the server provides closest_missile_speed (final summary), show it in the speed field
        if (data.closest_missile_speed !== undefined && data.closest_missile_speed !== null) {
            document.getElementById('speed-disp').innerText = Number(data.closest_missile_speed).toFixed(2);
        }

        mX.push(data.missile[0]);
        mY.push(data.missile[1]);
        mZ.push(data.missile[2]);

        tX.push(data.target[0]);
        tY.push(data.target[1]);
        tZ.push(data.target[2]);

        // Обновляем график
        Plotly.restyle('graph', {
            x: [mX, tX],
            y: [mY, tY],
            z: [mZ, tZ]
        }, [0, 1]);

        // Если было попадание — остановить симуляцию и разблокировать кнопку
        if (data.hit) {
            // On hit: stop simulation and keep extra rows hidden
            isRunning = false;
            document.getElementById('simulate-btn').disabled = false;
            document.getElementById('calculate-btn').disabled = false;
            try { socket.close(); } catch (e) {}
            document.getElementById('min-dist-row').style.display = 'none';
            document.getElementById('closest-time-row').style.display = 'none';
            // ensure speed shows current missile speed (if provided)
            if (data.missile_speed !== undefined) {
                document.getElementById('speed-disp').innerText = Number(data.missile_speed).toFixed(2);
            }
            console.log('Hit detected, simulation stopped.');
        }

        // If server ended simulation without hit, it sends a final summary with note
        if (data.note === 'simulation_finished_no_hit') {
            // On miss: show extra rows and set speed to closest_missile_speed if provided
            isRunning = false;
            document.getElementById('simulate-btn').disabled = false;
            document.getElementById('calculate-btn').disabled = false;
            try { socket.close(); } catch (e) {}
            console.log('Simulation finished without hit; final summary received.');
            if (data.min_distance !== undefined && data.min_distance !== null) {
                document.getElementById('min-dist-row').style.display = '';
            }
            if (data.closest_time !== undefined && data.closest_time !== null) {
                document.getElementById('closest-time-row').style.display = '';
            }
            if (data.closest_missile_speed !== undefined && data.closest_missile_speed !== null) {
                document.getElementById('speed-disp').innerText = Number(data.closest_missile_speed).toFixed(2);
            }
        }
    };

    socket.onclose = function() {
        isRunning = false;
        document.getElementById('simulate-btn').disabled = false;
        document.getElementById('calculate-btn').disabled = false;
        console.log("Simulation connection closed");
    };

    socket.onerror = function(error) {
        isRunning = false;
        document.getElementById('simulate-btn').disabled = false;
        document.getElementById('calculate-btn').disabled = false;
        console.log("Simulation WebSocket error:", error);
    };
}

// Сброс расчетов при изменении параметров
function resetCalculation() {
    calculatedTrajectory = null;
    hideSimulateButton();
}

wsInput.oninput = function() {
    updateInputs();
    resetCalculation();
};

wdInput.oninput = function() {
    updateInputs();
    resetCalculation();
};

document.getElementById('precision-select').onchange = resetCalculation;