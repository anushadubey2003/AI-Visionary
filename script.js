const video = document.getElementById('video');
const captureBtn = document.getElementById('capture');
const resultsDiv = document.getElementById('results');
const historyDiv = document.getElementById('history');
const themeToggle = document.getElementById('theme-toggle');

navigator.mediaDevices.getUserMedia({ video: true })
.then(stream => { video.srcObject = stream; })
.catch(err => { alert("Webcam access denied"); });

function toBase64(canvas) {
    return canvas.toDataURL("image/jpeg");
}

captureBtn.addEventListener('click', async () => {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    const imageBase64 = toBase64(canvas);

    const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageBase64 })
    });
    const data = await response.json();
    displayResults(data.predictions, data.history);
});

function displayResults(predictions, history) {
    resultsDiv.innerHTML = "<h3>Current Prediction:</h3>";
    predictions.forEach(pred => {
        const card = document.createElement('div');
        card.className = 'prediction-card';
        card.innerText = `Age: ${pred.age_range}, Gender: ${pred.gender}, Confidence: ${pred.confidence}`;
        resultsDiv.appendChild(card);
    });

    historyDiv.innerHTML = "<h3>Prediction History:</h3>";
    history.forEach(pred => {
        const card = document.createElement('div');
        card.className = 'prediction-card';
        card.innerText = `Age: ${pred.age_range}, Gender: ${pred.gender}, Confidence: ${pred.confidence}`;
        historyDiv.appendChild(card);
    });
}

themeToggle.addEventListener('click', () => {
    document.body.classList.toggle('dark');
});
