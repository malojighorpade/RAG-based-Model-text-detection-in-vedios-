import whisper
import os
import json
model = whisper.load_model("large-v2")
audios = os.listdir("audio_mp3")
for audio in audios:
    title=audio.replace(".mp3","")
    result = model.transcribe(audio=f"audio_mp3/{audio}",language="hindi",task="translate",word_timestamps=False)
    chunks=[]
    for segment in result["segments"]:
        chunks.append({"title":title,"start":segment["start"],"end":segment["end"],"text":segment["text"]})
    chunks_with_metadata={"text":result["text"],"chunks":chunks}
    with open(f"jsons/{audio}.json", "w") as f:
        json.dump(chunks_with_metadata, f)

