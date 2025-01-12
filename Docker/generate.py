from fastapi import FastAPI
from pydantic import BaseModel
from models import build_model
import torch
from kokoro import generate
from fastapi.responses import FileResponse

# Initialize FastAPI app
app = FastAPI()

# Load model and voicepack
device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL = build_model('kokoro-v0_19.pth', device)
VOICE_NAME = 'af'  # Default voice
VOICEPACK = torch.load(f'voices/{VOICE_NAME}.pt', weights_only=True).to(device)

# Request model for input validation
class TTSRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Welcome to the Kokoro Text-to-Speech API"}

@app.post("/synthesize/")
async def synthesize(request: TTSRequest):
    # Generate audio
    audio, _ = generate(MODEL, request.text, VOICEPACK, lang=VOICE_NAME[0])

    # Save audio file
    output_path = "output.wav"
    with open(output_path, "wb") as f:
        f.write(audio.tobytes())

    return FileResponse(output_path, media_type="audio/wav", filename="synthesized_audio.wav")
