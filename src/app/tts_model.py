import numpy as np
import torch
from espnet2.bin.tts_inference import Text2Speech

from src.app.xvector import XVector


class TTSModel:
    model = None
    xvector = None

    @classmethod
    def get_model(cls):
        if cls.model is None:
            print("Started TTS model loading...")

            xvector_ark_path = "/opt/ml/model/xvector/spk_xvector.ark"
            model_file = "/opt/ml/model/vits/model.pth"
            vocoder_file = "/opt/ml/model/vits"

            cls.model = Text2Speech.from_pretrained(
                model_file=model_file,
                vocoder_file=vocoder_file,
            )
            cls.xvector = XVector(ark_path=xvector_ark_path)

        return cls.model, cls.xvector

    @classmethod
    def predict(cls, speaker_id: int, text_ch: str) -> tuple[np.ndarray, int]:
        model, xvector = cls.get_model()

        spembs = xvector.get_spembs(speaker_id)

        with torch.no_grad():
            out = model(text_ch, spembs=spembs)
            wav = out["wav"]

        wav_series = wav.view(-1).cpu().numpy()

        return wav_series, model.fs
