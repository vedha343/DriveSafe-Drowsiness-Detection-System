// Function to check status every 500 milliseconds (half a second)
setInterval(function() {
    fetch('/status')
        .then(response => response.json())
        .then(data => {
            const statusBox = document.getElementById("status-indicator");
            const instruction = document.getElementById("instruction-text");
            const status = data.result;

            if (status === "Drowsy") {
                // Change UI to Red Alert
                statusBox.innerText = "DROWSY!";
                statusBox.className = "status-drowsy"; // Uses CSS class for red styling
                instruction.innerText = "WAKE UP! ALARM TRIGGERED";
                
                // Play the alarm sound
                playAlarm();
            } else if (status === "No Face") {
                 // Change UI to Yellow/Gray
                 statusBox.innerText = "No Face Detected";
                 statusBox.className = "status-warning";
                 instruction.innerText = "Please look at the camera.";
            } else {
                // Change UI to Green/Active
                statusBox.innerText = "Active";
                statusBox.className = "status-active";
                instruction.innerText = "Safe driving detected.";
            }
        })
        .catch(error => console.error('Error fetching status:', error));
}, 500);

// Audio Logic
// Note: Browsers sometimes block auto-play audio. 
// You might need to click anywhere on the page once to enable audio.
let alarmAudio = new Audio('/static/alarm.mp3');

function playAlarm() {
    // Only play if it's not already playing to avoid echo
    if (alarmAudio.paused) {
        alarmAudio.play().catch(e => console.log("Audio play failed (interact with page first):", e));
    }
}