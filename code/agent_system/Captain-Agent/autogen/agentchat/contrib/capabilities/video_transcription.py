import sys
import warnings
import whisper
from pathlib import Path
import yt_dlp
import subprocess
import torch
import shutil
import numpy as np
from IPython.display import display, Markdown, YouTubeVideo
from whisper.utils import format_timestamp, make_safe

"""
Env Installation:

pip install git+https://github.com/openai/whisper.git
pip install yt-dlp
"""

def video_transcription(url: str):
    """
    TODO: tool description
    """

    device = torch.device('cuda:0')
    print('Using device:', device, file=sys.stderr)

    video_path_local_list = []

    ydl_opts = {
        'format': 'm4a/bestaudio/best',
        'outtmpl': '%(id)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }]
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        error_code = ydl.download([url])
        list_video_info = [ydl.extract_info(url, download=False)]

    for video_info in list_video_info:
        video_path_local_list.append(Path(f"{video_info['id']}.wav"))

    for video_path_local in video_path_local_list:
        if video_path_local.suffix == ".mp4":
            video_path_local = video_path_local.with_suffix(".wav")
            result = subprocess.run(["ffmpeg", "-i", str(video_path_local.with_suffix(".mp4")), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(video_path_local)])

    Model = "medium"
    language = "Auto detection" 
    verbose = 'None' 
    task = 'transcribe'
    temperature = 0.15
    temperature_increment_on_fallback = 0.2
    best_of = 5
    beam_size = 8
    patience = 1.0
    length_penalty = -0.05
    suppress_tokens = "-1"
    initial_prompt = ""
    condition_on_previous_text = True
    fp16 = True
    compression_ratio_threshold = 2.4
    logprob_threshold = -1.0
    no_speech_threshold = 0.6

    verbose_lut = {
        'Live transcription': True,
        'Progress bar': False,
        'None': None
    }

    args = dict(
        language = (None if language == "Auto detection" else language),
        verbose = verbose_lut[verbose],
        task = task,
        temperature = temperature,
        temperature_increment_on_fallback = temperature_increment_on_fallback,
        best_of = best_of,
        beam_size = beam_size,
        patience=patience,
        length_penalty=(length_penalty if length_penalty>=0.0 else None),
        suppress_tokens=suppress_tokens,
        initial_prompt=(None if not initial_prompt else initial_prompt),
        condition_on_previous_text=condition_on_previous_text,
        fp16=fp16,
        compression_ratio_threshold=compression_ratio_threshold,
        logprob_threshold=logprob_threshold,
        no_speech_threshold=no_speech_threshold
    )

    whisper_model = whisper.load_model(Model)
    temperature = args.pop("temperature")
    temperature_increment_on_fallback = args.pop("temperature_increment_on_fallback")
    if temperature_increment_on_fallback is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback))
    else:
        temperature = [temperature]

    if Model.endswith(".en") and args["language"] not in {"en", "English"}:
        warnings.warn(f"{Model} is an English-only model but receipted '{args['language']}'; using English instead.")
        args["language"] = "en"

    for video_path_local in video_path_local_list:
        video_transcription = whisper.transcribe(
            whisper_model,
            str(video_path_local),
            temperature=temperature,
            **args,
        )
        results = []
        for segment in video_transcription['segments']:
            start, end, text = segment["start"], segment["end"], segment["text"]
            line = f"[{format_timestamp(start)} --> {format_timestamp(end)}] {text}"
            results.append(make_safe(line))
        
        return "\n".join(results)

if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=1htKBjuUWec"
    print(video_transcription(url))