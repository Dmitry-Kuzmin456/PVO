let socket;
let isRunning = false;

const layout = {
    title: 'Траектория полета',
    scene: {
        // Убираем жесткие границы, чтобы камера следила за объектами
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
    x: [2000], y: [5000], z: [3000], marker: {color:'blue', size:4}
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

function startSimulation() {
    if (isRunning) return;

    document.getElementById('start-btn').disabled = true;
    isRunning = true;

    // Очищаем графики
    Plotly.restyle('graph', {x: [[0]], y: [[0]], z: [[0]]}, [0]);
    Plotly.restyle('graph', {x: [[2000]], y: [[5000]], z: [[3000]]}, [1]);

    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    socket = new WebSocket(`${protocol}://${window.location.host}/ws/simulate`);

    // Буферы для накопления точек
    let mX = [0], mY = [0], mZ = [0];
    let tX = [2000], tY = [5000], tZ = [3000];

    socket.onopen = function() {
        socket.send(JSON.stringify({
                wind_speed: Number(wsInput.value),
                wind_dir: Number(wdInput.value),
                precision: document.getElementById('precision-select').value
            }));
    };

    socket.onmessage = function(event) {
        const data = JSON.parse(event.data);

        // time may be sent as formatted string from server; ensure two decimals
        const timeStr = (typeof data.time === 'string') ? data.time : Number(data.time).toFixed(2);
        document.getElementById('time-disp').innerText = timeStr;
        document.getElementById('alt-disp').innerText = Math.round(data.missile[2]);
        // format distance to two decimals
        document.getElementById('dist-disp').innerText = (typeof data.distance === 'number') ? data.distance.toFixed(2) : data.distance;
        // display missile speed if provided
        if (data.missile_speed !== undefined) {
            document.getElementById('speed-disp').innerText = Number(data.missile_speed).toFixed(2);
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
            isRunning = false;
            document.getElementById('start-btn').disabled = false;
            try { socket.close(); } catch (e) {}
            console.log('Hit detected, simulation stopped.');
        }
    };

    socket.onclose = function() {
        isRunning = false;
        document.getElementById('start-btn').disabled = false;
        console.log("Connection closed");
    };
}
