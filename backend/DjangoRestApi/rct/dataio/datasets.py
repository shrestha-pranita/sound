import numpy as np
import torch
import torchaudio


def to_mono(mixture, random_ch=False):

    if mixture.ndim > 1:  # multi channel
        if not random_ch:
            mixture = torch.mean(mixture, 0)
        else:  # randomly select one channel
            indx = np.random.randint(0, mixture.shape[0] - 1)
            mixture = mixture[indx]
    return mixture


def pad_audio(audio, target_len):
    if audio.shape[-1] < target_len:
        audio = torch.nn.functional.pad(
            audio, (0, target_len - audio.shape[-1]), mode="constant"
        )
        padded_indx = [target_len / len(audio)]
    else:
        padded_indx = [1.0]

    return audio, padded_indx


def read_audio(file, multisrc, random_channel, pad_to=None):
    mixture, fs = torchaudio.load(file)
    if not multisrc:
        mixture = to_mono(mixture, random_channel)

    if pad_to is not None:
        mixture, padded_indx = pad_audio(mixture, pad_to)
    else:
        padded_indx = [1.0]

    mixture = mixture.float()
    return mixture, padded_indx
