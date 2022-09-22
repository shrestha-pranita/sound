import os
import io
import torch
import numpy as np
from torch import torch
from torch import Tensor
from time import time
from pathlib import Path
import noisereduce as nr
import rx.operators as ops
import diart.sources as src
from rx.core import Observer
from scipy.io import wavfile
from datetime import datetime
import diart.operators as dops
from traceback import print_exc
from pyannote.database.util import load_rttm
from diart.pipelines import OnlineSpeakerDiarization
from typing import Union, Text, Optional, Tuple
from pyannote.core import Annotation, SlidingWindowFeature
from speechbrain.pretrained import VAD
#from hydra import compose, initialize
from rest_framework import status
from django.http import JsonResponse
import scipy.io.wavfile
from torch.nn import Module
import scipy.signal as sps




class RTTMWriter(Observer):
  def __init__(self, path: Union[Path, Text], patch_collar: float = 0.05):
    super().__init__()
    self.patch_collar = patch_collar
    self.path = Path(path)
    if self.path.exists():
      self.path.unlink()

  def patch_rttm(self):
    
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
VAD = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty",
                       savedir="pretrained_models/vad-crdnn-libriparty")

pipeline = OnlineSpeakerDiarization(
    step=0.5,
    latency=0.5,
    tau_active=0.555,
    rho_update=0.422,
    delta_new=1.517,
    gamma=3,
    beta=10,
    max_speakers=5
)



def predict_mul(request, filepath):
    save_root = '/'.join(filepath.split('/')[:-1]) + '/'
    if 'Records' not in os.listdir(save_root):
        os.makedirs(save_root + 'Records')
    if 'Diarization' not in os.listdir(save_root):
        os.makedirs(save_root + 'Diarization')
    response = JsonResponse({'status': 'fail', 'description': 'Server Issues'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    noise_presence = 'no'
    multi_speaker = 'no'
    base_name = filepath.split('/')[-1]
    current_recording = save_root + 'Records/' + base_name + '.wav'
    current_diarization = save_root + 'Diarization/' + base_name + '.txt'

    record_file = io.open(save_root+'Records/'+'00-RecordJournal.csv', "a", encoding="utf-8")
    try:
        mark = time()
        sample_rate, src_data = scipy.io.wavfile.read(filepath)
        if sample_rate != SAMPLE_RATE:
            number_of_samples = round(len(src_data) * float(SAMPLE_RATE) / sample_rate)
            src_data = sps.resample(src_data, number_of_samples)
        data = nr.reduce_noise(y=src_data, sr=SAMPLE_RATE)
        speech_probs = VAD.get_speech_prob_chunk(torch.tensor(data))
        activation_pass_values = VAD.apply_threshold(speech_probs).numpy()
        vad_prob = activation_pass_values.sum()/np.prod(activation_pass_values.shape)
        confidence = speech_probs.numpy().sum()/np.prod(activation_pass_values.shape)
        record_file.write(base_name+'.wav'+', {:.2f}'.format(vad_prob)+', {:.2f}'.format(confidence)+'\n')
        record_file.close()
        if vad_prob > 0.5:
            source_temp = Path(filepath).expanduser()
            audio_source = src.FileAudioSource(
                file=source_temp,
                uri=''.join(source_temp.name.split(".")[:-1]),
                reader=src.RegularAudioFileReader(
                    SAMPLE_RATE, 1.0, pipeline.step
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
                print(i)
                speakers.add(i.split(' ')[7])
            if len(speakers) > 1:
                multi_speaker = 'yes'
            print(multi_speaker)
            response = JsonResponse({'status': 'success', 'prediction': '',
                                'multi_speaker': multi_speaker, 'time_taken': str(time()-mark)[:4]+' Sec'}, status=status.HTTP_200_OK)
        else:
            response = JsonResponse({'status': 'success', 'prediction': '',
                                'multi_speaker': multi_speaker, 'time_taken': str(time()-mark)[:4]+' Sec'}, status=status.HTTP_200_OK)
            
        return response
    except:
        print("except")
        response = JsonResponse({'status': 'fail', 'description': 'Server Issues'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return response
