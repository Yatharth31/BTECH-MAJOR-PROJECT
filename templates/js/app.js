var gumStream;
var mediaRecorder;
var audioChunks = [];

var recordButton = document.getElementById("recordButton");
var stopButton = document.getElementById("stopButton");

recordButton.addEventListener("click", startRecording);
stopButton.addEventListener("click", stopRecording);

function startRecording() {
    recordButton.disabled = true;
    stopButton.disabled = false;

    var constraints = { audio: true, video: false };

    navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
        gumStream = stream;
        mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
        audioChunks = [];

        mediaRecorder.ondataavailable = function (e) {
            if (e.data.size > 0) {
                audioChunks.push(e.data);
            }
        };

        mediaRecorder.start();
    }).catch(function (err) {
        console.error("Error accessing microphone: ", err);
        recordButton.disabled = false;
        stopButton.disabled = true;
    });
}

function stopRecording() {
    stopButton.disabled = true;
    recordButton.disabled = false;

    mediaRecorder.stop();
    gumStream.getAudioTracks()[0].stop();

    mediaRecorder.onstop = function () {
        var blob = new Blob(audioChunks, { type: "audio/webm" });
        uploadAudio(blob);
    };
}

function uploadAudio(blob) {
    var formData = new FormData();
    formData.append("audio_file", blob, "recorded_audio.webm");

    fetch("/save_audio", {
        method: "POST",
        body: formData,
    })
        .then(response => response.json())
        .then(data => {
            console.log("Audio uploaded successfully:", data);
        })
        .catch(error => {
            console.error("Error uploading audio:", error);
        });
}
