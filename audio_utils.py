import os
from pydub import AudioSegment

def mp4_to_wav(mp4_path):
    wav_path = mp4_path.replace(".mp4", ".wav")
    audio = AudioSegment.from_file(mp4_path, format="mp4")
    audio.export(wav_path, format="wav")
    return wav_path

def split_audio(wav_path, chunk_length=5):
    audio = AudioSegment.from_wav(wav_path)
    chunks = []
    duration = len(audio) // 1000  # milliseconds → seconds
    os.makedirs("chunks", exist_ok=True)
    for i in range(0, duration, chunk_length):
        chunk = audio[i * 1000:(i + chunk_length) * 1000]
        chunk_path = f"chunks/chunk_{i//chunk_length}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks