import os
import subprocess

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from faster_whisper import WhisperModel
import torch
import uuid

from ray import serve

app = FastAPI()

class Whisper:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.compute_type = "float16" if torch.cuda.is_available() else "float32"
        self.model = WhisperModel('medium.en', device=self.device, compute_type=self.compute_type, local_files_only=True)
        command = "python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + \":\" + os.path.dirname(nvidia.cudnn.lib.__file__))'"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
        else:
            library_paths = result.stdout.strip()
            os.environ['LD_LIBRARY_PATH'] = library_paths
            print(f"LD_LIBRARY_PATH set to: {os.environ['LD_LIBRARY_PATH']}")

whisper = Whisper()

@app.post("/transcribe")
async def transcribe(audio_file: UploadFile = File(...)) -> JSONResponse:
    if audio_file.content_type not in ["audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp4"]:
        raise HTTPException(status_code=400, detail="Invalid audio file type")
    file_id = str(uuid.uuid4())
    temp_file_path = f"temp_{file_id}.wav"
    try:
        audio_data = await audio_file.read()
        with open(temp_file_path, "wb") as f:
            f.write(audio_data)
        transcription, _ = whisper.model.transcribe(temp_file_path)
        return JSONResponse([segment._asdict() for segment in transcription])

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 1, "num_gpus": .5})
@serve.ingress(app)
class FastAPIWrapper:
    pass

deployment = FastAPIWrapper.bind()