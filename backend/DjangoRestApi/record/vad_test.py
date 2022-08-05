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
from diart.pipelines import OnlineSpeakerDiarization, PipelineConfig
from typing import Union, Text, Optional, Tuple
from pyannote.core import Annotation, SlidingWindowFeature
from speechbrain.pretrained import VAD
#from hydra import compose, initialize
from rest_framework import status
from django.http import JsonResponse
import scipy.io.wavfile
from torch.nn import Module
import scipy.signal as sps

"""
from .snr.data import (
    NoisedAudPipeline,
    AudioPipeline
    )
from .snr.utils import load_model
"""


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

"""
def get_melkwargs() -> dict:
    return {
        'n_fft': hprams.data.n_fft,
        'win_length': hprams.data.win_length,
        'hop_length': hprams.data.hop_length
    }


def get_snr_params() -> dict:
    return {
        'sample_rate': hprams.data.sampling_rate,
        'win_length': hprams.data.win_length,
        'hop_length': hprams.data.hop_length,
        'min_snr': hprams.data.min_snr,
        'max_snr': hprams.data.max_snr
    }
"""
"""
class Predict:
    def __init__(
            self,
            #noised_pipeline: NoisedAudPipeline,
            model: Module,
            device: str
            ) -> None:
        #self.noised_pipeline = noised_pipeline
        self.model = model.to(device)
        self.device = device

    def predict(self, signal: Tensor):
        signal = self.noised_pipeline.run(signal)
        signal = signal.permute(0, 2, 1)
        return self.model(signal, torch.tensor([signal.shape[1]]))
"""
"""
def get_predictor() -> Predict:
    model = load_model(
        hprams.model, 
        hprams.checkpoint, 
        device=hprams.device
    )
    noised_pipeline = NoisedAudPipeline(
        hprams.data.sampling_rate,
        hprams.data.n_mfcc,
        get_melkwargs()
    )
    return Predict(
        noised_pipeline,
        model,
        hprams.device
    )
"""
SAMPLE_RATE = 16000
VAD = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty",
                       savedir="pretrained_models/vad-crdnn-libriparty")

#initialize(config_path="./snr")
#hprams = compose(config_name="configs")
#get_melkwargs()
#predictor = get_predictor()
#aud_pipeline = AudioPipeline(hprams.data.sampling_rate)
#noised_pipeline = NoisedAudPipeline(
 #       hprams.data.sampling_rate,
  #      hprams.data.n_mfcc,
   #     get_melkwargs()
#)
"""
config = PipelineConfig(
    step=0.5,
    latency=0.5,
    tau_active=0.555,
    rho_update=0.422,
    delta_new=1.517,
    gamma=3,
    beta=10,
    max_speakers=5
)
"""
"""
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
"""

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
    pipeline = OnlineSpeakerDiarization(config)
    save_root = '/'.join(filepath.split('/')[:-1]) + '/'
    if 'Records' not in os.listdir(save_root):
        os.makedirs(save_root + 'Records')
    if 'Diarization' not in os.listdir(save_root):
        os.makedirs(save_root + 'Diarization')
   
    #response = jsonify({'status': 'fail', 'description': 'no audio data detected!!'})
    response = JsonResponse({'status': 'fail', 'description': 'Server Issues'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    noise_presence = 'no'
    multi_speaker = 'no'
    base_name = filepath.split('/')[-1]
    #base_name =  str(datetime.now()).replace(' ', '_')
    current_recording = save_root + 'Records/' + base_name + '.wav'
    current_diarization = save_root + 'Diarization/' + base_name + '.txt'

    record_file = io.open(save_root+'Records/'+'00-RecordJournal.csv', "a", encoding="utf-8")
    try:
        mark = time()
        #content = np.array(request.get_json()['data'], np.int16)
        sample_rate, src_data = scipy.io.wavfile.read(filepath)
        if sample_rate != SAMPLE_RATE:
            number_of_samples = round(len(src_data) * float(SAMPLE_RATE) / sample_rate)
            src_data = sps.resample(src_data, number_of_samples)
        data = nr.reduce_noise(y=src_data, sr=SAMPLE_RATE)
        #content = np.array(request.data["data"], np.int16)
        #wavfile.write(current_recording, SAMPLE_RATE, content)
        #sound = aud_pipeline.run(current_recording)
        #preds = predictor.predict(sound).squeeze()
        #if Tensor.sum(preds)/preds.shape[0] < 3:
            #noise_presence = 'yes'
        
        #data = nr.reduce_noise(y=content, sr=SAMPLE_RATE)
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
            #response = jsonify({'status': 'success', 'prediction': 'Voice Detected', 'noise_presence': noise_presence,
                                #'multi_speaker': multi_speaker, 'time_taken': str(time()-mark)[:4]+' Sec'})
        else:
            response = JsonResponse({'status': 'success', 'prediction': '',
                                'multi_speaker': multi_speaker, 'time_taken': str(time()-mark)[:4]+' Sec'}, status=status.HTTP_200_OK)
            #response = jsonify({'status': 'success', 'prediction': 'Voice Not Detected', 'noise_presence': noise_presence,
                                #'multi_speaker': multi_speaker, 'time_taken': str(time()-mark)[:4]+' Sec', 'confidence': confidence})
        return response
    except:
        print("except")
        #response = jsonify({'status': 'fail', 'description': 'Detection Failed!!'})
        response = JsonResponse({'status': 'fail', 'description': 'Server Issues'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return response
