from gtts import gTTS
import os

# The text you want the alarm to speak
text = "Alert! Drowsiness detected. Please wake up!"

# Generate the audio
tts = gTTS(text=text, lang='en')

# Save it directly to the static folder
print("Generating alarm.mp3...")
tts.save("static/alarm.mp3")
print("Success! 'alarm.mp3' has been created in your static folder.")