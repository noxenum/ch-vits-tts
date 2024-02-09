import kaldiio
import numpy as np


class XVector:
    def __init__(self, ark_path: str):
        self.xvectors = {k: v for k, v in kaldiio.load_ark(ark_path)}
        self.spks = list(self.xvectors.keys())

    def get_spembs_by_name(self, speaker_name):
        return self.xvectors[speaker_name]

    def get_spembs(self, speaker_id: int):
        spk = self.spks[speaker_id]
        return self.xvectors[spk]

    def get_random_speaker(self):
        speaker_id = np.random.randint(0, len(self.spks))
        return self.get_spembs(speaker_id)

    def get_speaker_name(self, speaker_id: int):
        return self.spks[speaker_id]
