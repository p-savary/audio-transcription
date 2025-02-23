from pydub import AudioSegment
import subprocess
import os

DEVICE = os.getenv("DEVICE")


def isolate_voices(file_paths):
    for index in range(len(file_paths)):
        chunk_length_ms = 100
        chunked = []
        for file in file_paths:
            audio = AudioSegment.from_file(file)
            chunked.append([audio[i : i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)])

        processed_chunks = [
            filter_nondominant_voice([chunks[i] for chunks in chunked], index) for i in range(len(chunked[0]))
        ]

        processed_audio = sum(processed_chunks)
        processed_audio.export(file_paths[index])


def filter_nondominant_voice(segments, index):
    value = segments[index].dBFS
    for i, segment in enumerate(segments):
        if i == index:
            continue
        if segment.dBFS > value:
            return segment - 100
    return segments[index]


def get_length(filename):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            filename,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return float(result.stdout)


def time_estimate(filename, online=True):
    try:
        # For now, we don't predict the wait time for zipped files in the queue.
        if filename[-4:] == ".zip":
            return 1, 1
        run_time = get_length(filename)
        if online:
            if DEVICE == "mps":
                return run_time / 5, run_time
            else:
                return run_time / 10, run_time
        else:
            if DEVICE == "mps":
                return run_time / 3, run_time
            else:
                return run_time / 6, run_time
    except Exception as e:
        print(e)
        return -1, -1
