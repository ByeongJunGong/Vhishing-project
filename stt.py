import whisper

def transcribe_with_segments(audio_path):
    model = whisper.load_model("large")  
    result = model.transcribe(audio_path, language="ko", word_timestamps=False)
    return result['segments']
