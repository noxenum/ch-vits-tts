import base64
import io

from fastapi import FastAPI
from pydantic import BaseModel
from scipy.io.wavfile import write

from src.app.translation_model import TranslationModel
from src.app.tts_model import TTSModel

app = FastAPI()

id_to_dialect = {
    0: "ag",
    1: "be",
    2: "bs",
    3: "gr",
    4: "lu",
    5: "sg",
    6: "vs",
    7: "zh",
}


class TTSInputModel(BaseModel):
    text_de: str
    voice_id: int


@app.get("/ping")
async def ping():
    return {"status_code": 200}


@app.post("/invocations", status_code=200)
async def predict(tts_input: TTSInputModel):
    dialect = id_to_dialect[tts_input.voice_id]

    text_ch = TranslationModel.predict(
        dialect=dialect, text_de=tts_input.text_de, beam_size=1
    )

    wav, sr = TTSModel.predict(speaker_id=tts_input.voice_id, text_ch=text_ch)
    bytes_wav = bytes()
    byte_io = io.BytesIO(bytes_wav)
    write(byte_io, sr, wav)
    wav_bytes = byte_io.read()

    audio_data = base64.b64encode(wav_bytes).decode("UTF-8")

    return {"status_code": 200, "audio": audio_data, "text_ch": text_ch}
