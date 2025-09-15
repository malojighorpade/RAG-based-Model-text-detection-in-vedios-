import os
import subprocess

# Converts the videos to mp3
files = os.listdir("videos")


for file in files:

    

    if file.endswith(".mp4"):
        name, ext = os.path.splitext(file)
        subprocess.run(['ffmpeg', '-i', f"videos/{file}", f"audio_mp3/{name}.mp3"])


