import os
import io
import torch
import librosa
import numpy as np
from time import time
from pathlib import Path
import noisereduce as nr
import rx.operators as ops
import diart.sources as src
from rx.core import Observer
import scipy.signal as sps
from scipy.io import wavfile
import diart.operators as dops
from traceback import print_exc
from pyannote.database.util import load_rttm
from rest_framework import status
from django.http import JsonResponse
from diart.pipelines import OnlineSpeakerDiarization
from typing import Union, Text, Optional, Tuple
from pyannote.core import Annotation, SlidingWindowFeature
from speechbrain.pretrained import VAD
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Activation
import pandas as pd


class RTTMWriter(Observer):
  def __init__(self, path: Union[Path, Text], patch_collar: float = 0.05):
    super().__init__()
    self.patch_collar = patch_collar
    self.path = Path(path)
    if self.path.exists():
      self.path.unlink()

  def patch_rttm(self):
    """Stitch same-speaker turns that are close to each other"""
    loaded_rttm = list(load_rttm(self.path).values())
    if len(loaded_rttm) != 0:
      annotation = loaded_rttm[0]
      with open(self.path, 'w') as file:
        annotation.support(self.patch_collar).write_rttm(file)

  def on_next(self, value: Tuple[Annotation, Optional[SlidingWindowFeature]]):
    with open(self.path, 'a') as file:
      value[0].write_rttm(file)

  def on_error(self, error: Exception):
    try:
      self.patch_rttm()
    except Exception:
      print("Error while patching RTTM file:")
      print_exc()
      exit(1)

  def on_completed(self):
    self.patch_rttm()

def get_client_ip(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

SAMPLE_RATE = 16000
#VAD = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="pretrained_models/vad-crdnn-libriparty", overrides={"sample_rate": SAMPLE_RATE})
                       #savedir="pretrained_models/vad-crdnn-libriparty")
# VAD = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="pretrained_models/vad-crdnn-libriparty")

pipeline = OnlineSpeakerDiarization(
    step=0.5,
    latency=0.5,
    tau_active=0.5,
    rho_update=0.3,
    delta_new=1,
    gamma=3,
    beta=10,
    max_speakers=5,
)

def audio_to_spectrogram(test_audio_filename):
    audio, sample_rate = librosa.load(test_audio_filename, res_type='kaiser_fast')
    spectrogram_test = librosa.feature.melspectrogram(audio,sample_rate)
    mfccs_scaled_features = np.mean(spectrogram_test,axis=0)
    print(spectrogram_test.shape)
    spectrogram_test = np.reshape(spectrogram_test,(1,128,44,1))
    #spectrogram_test = np.reshape(spectrogram_test,(1,128,216,1))
    return spectrogram_test

def env_classification(filepath):
  esc50_csv = './ESC-50/meta/esc50.csv'
  pd_data = pd.read_csv(esc50_csv)
  my_classes=list(pd_data["category"].unique())


  keras_model_path = './models'
  reloaded_model = keras.models.load_model(keras_model_path)
  reloaded_model.predict(audio_to_spectrogram(filepath))
  predicted_label = reloaded_model.predict(audio_to_spectrogram(filepath))
  print(f"The predicted label is {predicted_label[0]}")
  predicted_class = my_classes[tf.argmax(predicted_label[0])]
  print(f"The predicted class is {predicted_class}")
  probs = predicted_label.max(1)
  print(f"The probability is {probs}")

def predict(request, filepath):
  if request.method == 'POST' and filepath:
    save_root = '/'.join(filepath.split('/')[:-1]) + '/'
    try:
      if str('Diarization') not in os.listdir(save_root):
        os.makedirs(save_root + 'Diarization')
    except:
      response = JsonResponse({'status': 'fail', 'description': 'Server Issues'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
      print('Directory Related Issues')
      return response
    noise_presence = 'no'
    multi_speaker = 'no'
    #base_name = filepath.split('/')[-1]
    base_name = Path(filepath).stem
    current_diarization = save_root + 'Diarization/' + base_name + '.txt'
    try:
      mark = time()
      sample_rate, src_data = wavfile.read(filepath)
      if sample_rate != SAMPLE_RATE:
        number_of_samples = round(len(src_data) * float(SAMPLE_RATE) / sample_rate)
        src_data = sps.resample(src_data, number_of_samples)
      data = nr.reduce_noise(y=src_data, sr=SAMPLE_RATE)
      sample_float = (src_data/32768).astype(np.float32)
      flatness = librosa.feature.spectral_flatness(y=sample_float)

      env_classification(filepath)
      if np.sum(flatness)/np.prod(flatness.shape) > 0.3:
        noise_presence = 'yes'
      # speech_probs = VAD.get_speech_prob_chunk(torch.tensor(data))
      # activation_pass_values = VAD.apply_threshold(speech_probs).numpy()
      # vad_prob = activation_pass_values.sum()/np.prod(activation_pass_values.shape)
      # confidence = speech_probs.numpy().sum()/np.prod(activation_pass_values.shape)
      # if vad_prob > 0.5:
      source_temp = Path(filepath).expanduser()
      audio_source = src.FileAudioSource(
          file=source_temp,
          uri=''.join(source_temp.name.split(".")[:-1]),
          reader=src.RegularAudioFileReader(
              sample_rate, 1.0, pipeline.step
          ),
      )
      pipeline.from_source(audio_source).pipe(
          ops.do(RTTMWriter(path=current_diarization)),
          dops.buffer_output(
              duration=pipeline.duration,
              step=pipeline.step,
              latency=pipeline.latency,
              sample_rate=audio_source.sample_rate
          ),
      ).subscribe()
      audio_source.read()
      speakers = set()
      for i in open(current_diarization).readlines():
        speakers.add(i.split(' ')[7])
      if len(speakers) > 0:
        multi_speaker = 'yes'
      response = JsonResponse({'status': 'success', 'prediction': '', 'noise_presence': noise_presence,
                            'multi_speaker': multi_speaker, 'time_taken': str(time()-mark)[:4]+' Sec'}, status=status.HTTP_200_OK)
      os.remove(current_diarization)
      os.remove(filepath)
      return response
    except:
      response = JsonResponse({'status': 'fail', 'description': 'Detection Failed!!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
      os.remove(current_diarization)
      os.remove(filepath)
      return response
