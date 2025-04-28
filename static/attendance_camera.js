// attendance_camera.js - Camera UI and AJAX for attendance
let videoStream = null;

function showStatus(msg, type = 'info') {
    const statusDiv = document.getElementById('attendance-status');
    statusDiv.textContent = msg;
    statusDiv.className = 'alert alert-' + type;
    statusDiv.style.display = 'block';
}

async function startCamera() {
    const video = document.getElementById('attendance-video');
    try {
        videoStream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = videoStream;
        video.play();
        showStatus('Camera started', 'info');
    } catch (err) {
        showStatus('Unable to access camera: ' + err.message, 'danger');
    }
}

function stopCamera() {
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
    }
}

function captureImage() {
    const video = document.getElementById('attendance-video');
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg');
}

async function sendImageForRecognition(courseId) {
    showStatus('Recognizing face...', 'info');
    const imageData = captureImage();
    try {
        const res = await fetch('/api/recognize_face', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': document.querySelector('input[name="csrf_token"]').value
            },
            body: JSON.stringify({
                image: imageData,
                course_id: courseId
            })
        });
        const result = await res.json();
        if (result.status === 'success') {
            const recognizedNameDiv = document.getElementById('recognized-name');
            if (Array.isArray(result.recognized)) {
                if (result.recognized.length === 0) {
                    recognizedNameDiv.textContent = 'No faces recognized.';
                } else {
                    recognizedNameDiv.innerHTML = result.recognized.map(person => `${person.name} ${person.roll !== '-' ? `(Roll: ${person.roll})` : ''}`).join('<br>');
                }
            } else {
                recognizedNameDiv.textContent = result.name + (result.roll !== '-' ? ` (Roll: ${result.roll})` : '');
            }
            showStatus('Recognized faces', 'success');
        } else {
            showStatus(result.message || 'Recognition failed.', 'warning');
            document.getElementById('recognized-name').textContent = 'Unknown';
        }
    } catch (err) {
        showStatus('Error: ' + err.message, 'danger');
        document.getElementById('recognized-name').textContent = 'Error';
    }
}

document.addEventListener('DOMContentLoaded', function() {
    const startBtn = document.getElementById('start-camera-btn');
    const snapBtn = document.getElementById('snap-btn');
    const stopBtn = document.getElementById('stop-camera-btn');
    const courseSelect = document.getElementById('attendance-course-id');

    if (startBtn) startBtn.onclick = startCamera;
    if (stopBtn) stopBtn.onclick = stopCamera;
    if (snapBtn) snapBtn.onclick = function() {
        sendImageForRecognition(courseSelect.value);
    };
});
