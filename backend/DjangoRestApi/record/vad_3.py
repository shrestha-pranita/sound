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

    
    SPEC_H = 128
    SPEC_W = 216
    
    
    spectrogram_test = np.resize(spectrogram_test,(1,SPEC_H, SPEC_W,1))

    return spectrogram_test

def env_classification(filepath):
  

  low = ['dog', 'rooster', 'pig', 'cow', 'frog', 'cat', 'hen', 'insects', 'sheep', 'crow', 'rain', 'sea_waves',
'crackling_fire', 'crickets', 'chirping_birds', 'water_drops', 'wind', 'thunderstorm', 'crying_baby', 'sneezing', 'breathing'
'coughing', 'brushing_teeth', 'snoring', 'drinking_sipping', 'mouse_click', 'keyboard_typing', 'can_opening', 'washing_machine',
'vacuum_cleaner', 'clock_alarm', 'clock_tick', 'glass_breaking', 'helicopter', 'chain_saw', 'siren', 'car_horn', 'engine', 'train',
'church_bells', 'airplane', 'crackers', 'hand_saw', 'drilling', 'engine_idling', 'gun_shot', ' jackhammer', 'siren', 'street_music',
'children_playing', 'air_conditioner']

  medium = ['pouring_water', 'toilet_flush', 'laughing']

  high = ['footsteps', 'door_knock', 'door_wood_creaks']

  esc50_csv = './ESC-50/meta/esc50.csv'
  pd_data = pd.read_csv(esc50_csv)
  my_classes=list(pd_data["category"].unique())


  keras_model_path = './models'
  reloaded_model = keras.models.load_model(keras_model_path)
  spec = audio_to_spectrogram(filepath)
  prediction = list(reloaded_model.predict(spec).flatten())
  print(max(prediction))
  print('\nPredicted class:',my_classes[prediction.index(max(prediction))])
  predicted_class = my_classes[prediction.index(max(prediction))]
  probs = prediction.max(1)
  

  noise_level = "low"
  if predicted_class in low:
    noise_level = "low"
  elif predicted_class in medium:
    noise_level = "medium"
  else:
    noise_level = "high"
 

  return predicted_class, probs, noise_level

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

      predicted_class, probs, noise_level = env_classification(filepath)
      print(predicted_class)
      try:
        speech_probs = VAD.get_speech_prob_chunk(torch.tensor(data))
      except:
        print("no")
      print(speech_probs)
      print("What")
      activation_pass_values = VAD.apply_threshold(speech_probs).numpy()
      vad_prob = activation_pass_values.sum()/np.prod(activation_pass_values.shape)
      confidence = speech_probs.numpy().sum()/np.prod(activation_pass_values.shape)
      print("herrtre")
      speech_detection = 'no'
      if vad_prob > 0.5:
        speech_detection = 'yes'

      print(vad_prob)
      response = JsonResponse({'status': 'success', 'prediction': '', 'noise_presence': 'yes', 'multi_speaker': 'yes', 
      'noise_level': noise_level, 'speech_detection': speech_detection, 'time_taken': str(time()-mark)[:4]+' Sec'}, status=status.HTTP_200_OK)
     
    except:
      response = JsonResponse({'status': 'success', 'prediction': '', 'noise_presence': 'yes',
                            'multi_speaker': 'yes', 'noise_level':'low' ,'time_taken': str(time()-mark)[:4]+' Sec'}, status=status.HTTP_200_OK)
    return response
