import os
import os.path
import shutil
import time
import fnmatch
import types
import ffmpeg
import torch
import whisperx
import zipfile

from os.path import isfile
from dotenv import load_dotenv
from pyannote.audio import Pipeline

from src.viewer import create_viewer
from src.srt import create_srt
from src.transcription import transcribe, get_prompt
from src.util import time_estimate


load_dotenv()

ONLINE = os.getenv("ONLINE") == "True"
DEVICE = os.getenv("DEVICE")
ROOT = os.getenv("ROOT")
WINDOWS = os.getenv("WINDOWS") == "True"
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))

if WINDOWS:
    os.environ["PATH"] += os.pathsep + "ffmpeg/bin"
    os.environ["PATH"] += os.pathsep + "ffmpeg"
    os.environ["PYANNOTE_CACHE"] = ROOT + "/models"
    os.environ["HF_HOME"] = ROOT + "/models"


def report_error(file_name, file_name_error, user_id, text=""):
    print(text)
    if not os.path.exists(ROOT + "data/error/" + user_id):
        os.makedirs(ROOT + "data/error/" + user_id)
    with open(file_name_error + ".txt", "w") as f:
        f.write(text)
    shutil.move(file_name, file_name_error)


def oldest_file(folder):
    matches = []
    times = []
    for root, _, filenames in os.walk(folder):
        for filename in fnmatch.filter(filenames, "*.*"):
            file_path = os.path.join(root, filename)
            matches.append(file_path)
            times.append(os.path.getmtime(file_path))

    return [m for _, m in sorted(zip(times, matches))]


def transcribe_file(file_name, multi_mode=False):
    data = None
    estimated_time = 0

    file = os.path.basename(file_name)
    user_id = os.path.normpath(file_name).split(os.path.sep)[-2]
    file_name_error = ROOT + "data/error/" + user_id + "/" + file
    file_name_out = ROOT + "data/out/" + user_id + "/" + file + ".mp4"
    progress_file_name = ""

    try:
        if (
            not multi_mode
            and os.path.exists(ROOT + "data/worker/" + user_id)
            and os.path.isdir(ROOT + "data/worker/" + user_id)
        ):
            shutil.rmtree(ROOT + "data/worker/" + user_id)
    except:
        # we probably don't have enough permission in ROOT/data/
        print("Could not remove folder: " + ROOT + "data/worker/" + user_id)

    try:
        if not multi_mode and not os.path.exists(ROOT + "data/out/" + user_id):
            os.makedirs(ROOT + "data/out/" + user_id)
    except:
        # we probably don't have enough permission in ROOT/data/
        print("Could not create output folder for user_id: " + user_id)
        return data, estimated_time, progress_file_name

    # estimate run time: length of audio divided by 13
    try:
        time.sleep(2)
        estimated_time, run_time = time_estimate(file_name, ONLINE)
        if run_time == -1:
            report_error(file_name, file_name_error, user_id, "Datei konnte nicht gelesen werden")
            return data, estimated_time, progress_file_name

    except Exception as e:
        print(e)
        report_error(file_name, file_name_error, user_id, "Datei konnte nicht gelesen werden")
        return data, estimated_time, progress_file_name

    if not multi_mode:
        try:
            if not os.path.exists(ROOT + "data/worker/" + user_id):
                os.makedirs(ROOT + "data/worker/" + user_id)
            progress_file_name = (
                ROOT + "data/worker/" + user_id + "/" + str(estimated_time) + "_" + str(time.time()) + "_" + file
            )
            with open(progress_file_name, "w") as f:
                f.write("")
        except:
            # we probably don't have enough permission in ROOT/data/
            print("Could not create progress_file")

    # the file most likely does not have a (valid) audio stream
    if not ffmpeg.probe(file_name, select_streams="a")["streams"]:
        report_error(file_name, file_name_error, user_id, "Die Tonspur der Datei konnte nicht gelesen werden")
        return data, estimated_time, progress_file_name

    if not multi_mode:
        # -y automatically replaces file in destination if it already exists.
        exit_status = os.system(
            f'ffmpeg -y -i "{file_name}" -filter:v scale=320:-2 -af "lowpass=3000,highpass=200" "{file_name_out}"'
        )
        if exit_status == 256:
            exit_status = os.system(
                f'ffmpeg -y -i "{file_name}" -c:v copy -af "lowpass=3000,highpass=200" "{file_name_out}"'
            )
        if not exit_status == 0:
            file_name_out = file_name
            print("Exit status: " + str(exit_status))
    else:
        file_name_out = file_name

    try:
        hotwords = []
        if isfile(ROOT + "data/in/" + user_id + "/hotwords.txt"):
            with open(ROOT + "data/in/" + user_id + "/hotwords.txt", "r") as h:
                hotwords = h.read().split("\n")
        data = transcribe(file_name_out, model, diarize_model, DEVICE, None, add_language=True, hotwords=hotwords)
    except Exception as e:
        report_error(file_name, file_name_error, user_id, "Transkription fehlgeschlagen")
        print(e)

    return data, estimated_time, progress_file_name


if __name__ == "__main__":
    if DEVICE == "cpu":
        compute_type = "float32"
    else:
        compute_type = "float16"

    if ONLINE:
        model = whisperx.load_model("large-v3", DEVICE, compute_type=compute_type)
    else:
        model = whisperx.load_model(
            "large-v3",
            DEVICE,
            compute_type=compute_type,
            download_root="models/whisperx",
        )

    model.model.get_prompt = types.MethodType(get_prompt, model.model)
    diarize_model = Pipeline.from_pretrained(
        "pyannote/speaker-diarization", use_auth_token=os.getenv("HF_AUTH_TOKEN")
    ).to(torch.device(DEVICE))

    for directory in ["data/in/", "data/out/", "data/error/", "data/worker/"]:
        if not os.path.exists(ROOT + directory):
            os.makedirs(ROOT + directory)

    print(
        """This transcription software (the Software) incorporates the open-source model Whisper Large v3 (the Model) and has been developed according to and with the intent to be used under Swiss law. Please be aware that the EU Artificial Intelligence Act (EU AI Act) may, under certain circumstances, be applicable to your use of the Software. You are solely responsible for ensuring that your use of the Software as well as of the underlying Model complies with all applicable local, national and international laws and regulations. By using this Software, you acknowledge and agree (a) that it is your responsibility to assess which laws and regulations, in particular regarding the use of AI technologies, are applicable to your intended use and to comply therewith, and (b) that you will hold us harmless from any action, claims, liability or loss in respect of your use of the Software."""
    )
    print("worker ready")
    while True:
        try:
            files_sorted_by_date = oldest_file(ROOT + "data/in/")
        except:
            pass

        for file_name in files_sorted_by_date:
            multi_mode = False

            file = os.path.basename(file_name)
            user_id = os.path.normpath(file_name).split(os.path.sep)[-2]

            file_name_error = ROOT + "data/error/" + user_id + "/" + file
            file_name_out = ROOT + "data/out/" + user_id + "/" + file + ".mp4"
            file_name_viewer = ROOT + "data/out/" + user_id + "/" + file + ".html"
            file_name_srt = ROOT + "data/out/" + user_id + "/" + file + ".srt"

            # Skip all files that have already been transcribed.
            if (not isfile(file_name) or isfile(file_name_viewer)) or file == "hotwords.txt":
                continue

            # Transcribe all zipped files and combine their audio
            if file_name[-4:] == ".zip":
                try:
                    shutil.rmtree(ROOT + "data/worker/zip", ignore_errors=True)
                    os.makedirs(ROOT + "data/worker/zip")
                    with zipfile.ZipFile(file_name, "r") as zip_ref:
                        zip_ref.extractall(ROOT + "data/worker/zip")
                    multi_mode = True
                    data_parts = []
                    estimated_time = 0
                    data = []
                    file_parts = []
                    for root, _, filenames in os.walk(ROOT + "data/worker/zip"):
                        for filename in fnmatch.filter(filenames, "*.*"):
                            estimated_time_part, run_time = time_estimate(os.path.join(root, filename), ONLINE)
                            estimated_time += estimated_time_part
                        progress_file_name = (
                            ROOT
                            + "data/worker/"
                            + user_id
                            + "/"
                            + str(estimated_time)
                            + "_"
                            + str(time.time())
                            + "_"
                            + file
                        )

                        with open(progress_file_name, "w") as f:
                            f.write("")

                        for filename in fnmatch.filter(filenames, "*.*"):
                            file_parts.append(' -i "' + str(os.path.join(root, filename)) + '"')
                            data_part, _, _ = transcribe_file(os.path.join(root, filename), multi_mode)
                            data_parts.append(data_part)

                    earliest_index = 0
                    while True and earliest_index >= 0:
                        earliest_start = -1
                        earliest_index = -1
                        for i in range(len(data_parts)):
                            if len(data_parts[i]) > 0 and (
                                earliest_start == -1 or data_parts[i][0]["start"] < earliest_start
                            ):
                                earliest_start = data_parts[i][0]["start"]
                                earliest_index = i
                        data.append(data_parts[earliest_index][0])
                        data_parts[earliest_index] = data_parts[earliest_index][1:]

                        if sum([len(part) for part in data_parts]) == 0:
                            break

                    if not isfile(file_name_out):
                        os.system(
                            "ffmpeg "
                            + "".join(file_parts)
                            + " -filter_complex amix=inputs="
                            + str(len(file_parts))
                            + ":duration=first "
                            + ROOT
                            + "data/worker/zip/tmp.mp4"
                        )

                        # -y automatically replaces file in destination if it already exists.
                        exit_status = os.system(
                            'ffmpeg -y -i "'
                            + ROOT
                            + "data/worker/zip/tmp.mp4"
                            + f'" -filter:v scale=320:-2 -af "lowpass=3000,highpass=200" "{file_name_out}"'
                        )
                        if exit_status == 256:
                            exit_status = os.system(
                                'ffmpeg -y -i "'
                                + ROOT
                                + "data/worker/zip/tmp.mp4"
                                + f'" -c:v copy -af "lowpass=3000,highpass=200" "{file_name_out}"'
                            )
                        shutil.rmtree(ROOT + "data/worker/zip", ignore_errors=True)
                        if not exit_status == 0:
                            print("Exit status: " + str(exit_status))
                except Exception as e:
                    report_error(file_name, file_name_error, user_id, "Transkription fehlgeschlagen")
                    print(e)
            else:
                data, estimated_time, progress_file_name = transcribe_file(file_name)

            if data is None:
                continue
            try:
                srt = create_srt(data)
                viewer = create_viewer(data, file_name_out, True, False, ROOT)

                with open(file_name_viewer, "w", encoding="utf-8") as f:
                    f.write(viewer)
                with open(file_name_srt, "w", encoding="utf-8") as f:
                    f.write(srt)

                print("Estimated Time " + str(estimated_time))

            except Exception as e:
                report_error(file_name, file_name_error, user_id, "Fehler beim Erstellen des Editors")
                print(e)

            if os.path.exists(progress_file_name):
                os.remove(progress_file_name)

            break

        time.sleep(1)
