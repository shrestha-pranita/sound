import os
from copy import deepcopy

import noisereduce as nr
import pytorch_lightning as pl
import torch
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

from dataio.datasets import read_audio
from local.utils import batched_decode_preds
from utils.scaler import TorchScaler


class SEDTask4_2021(pl.LightningModule):
    def __init__(
        self,
        hparams,
        sed_student,
        encoder,
        device,
    ):
        super(SEDTask4_2021, self).__init__()
        self.hparams.update(hparams) 
        #self.hparams = hparams
        self.encoder = encoder
        self.sed_student = sed_student
        self.sed_teacher = deepcopy(sed_student)

        feat_params = self.hparams["feats"]
        self.mel_spec = MelSpectrogram(
            sample_rate=feat_params["sample_rate"],
            n_fft=feat_params["n_window"],
            win_length=feat_params["n_window"],
            hop_length=feat_params["hop_length"],
            f_min=feat_params["f_min"],
            f_max=feat_params["f_max"],
            n_mels=feat_params["n_mels"],
            window_fn=torch.hamming_window,
            wkwargs={"periodic": False},
            power=1,
        ).to(device)

        for param in self.sed_teacher.parameters():
            param.detach_()

        self.scaler = self._init_scaler()

    def _init_scaler(self):
        if self.hparams["scaler"]["statistic"] == "instance":
            scaler = TorchScaler(
                "instance",
                self.hparams["scaler"]["normtype"],
                self.hparams["scaler"]["dims"],
            )

            return scaler

        if self.hparams["scaler"]["savepath"] is not None:
            if os.path.exists(self.hparams["scaler"]["savepath"]):
                scaler = torch.load(self.hparams["scaler"]["savepath"])
                print(
                    "Loaded Scaler from previous checkpoint from {}".format(
                        self.hparams["scaler"]["savepath"]
                    )
                )
                return scaler

    def take_log(self, mels):
        amp_to_db = AmplitudeToDB(stype="amplitude")
        amp_to_db.amin = 1e-5  # amin= 1e-5 as in librosa
        return amp_to_db(mels).clamp(min=-50, max=80)  # clamp to reproduce old code

    def detect(self, mel_feats, model):
        return model(self.scaler(self.take_log(mel_feats)))

    def predict(self, audio_path):
        audio, _ = read_audio(
            audio_path,
            multisrc=False,
            random_channel=False,  # pad_to=2 * 16000
        )
        audio = torch.tensor(nr.reduce_noise(y=audio, sr=16000))
        audio = torch.reshape(
            audio.to(device="cuda" if torch.cuda.is_available() else "cpu"),
            (1, audio.shape[0]),
        )

        mels = self.mel_spec(audio)
        strong_preds, _ = self.detect(mels, self.sed_student)

        decoded_strong = batched_decode_preds(
            strong_preds,
            [audio_path],
            self.encoder,
            median_filter=1,
            thresholds=[0.8],
        )
        return decoded_strong[0.8][decoded_strong[0.8]["event_label"] == "Speech"]
