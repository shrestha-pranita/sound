from django.shortcuts import render
from rest_framework.decorators import api_view
from django.http import JsonResponse, HttpResponse
from rest_framework import status, response
import os
from datetime import datetime
from datetime import date
from scipy.io import wavfile
import json
import numpy as np
import torch
from utils_vad import get_speech_timestamps, read_audio
#import pyaudio
from time import time

import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers import Conv2D, GlobalAveragePooling2D
from keras.layers import BatchNormalization, Add
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
import pickle
import librosa
import torch
import yaml
from dataio.datasets import read_audio
from local.utils import batched_decode_preds
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
import noisereduce as nr
#from utils.scaler import TorchScaler
#from utils.encoder import ManyHotEncoder

#from model import SEDTask4_2021
from nnet.CRNN import CRNN
import os
from speechbrain.pretrained import SpeakerRecognition
#from .vad import predict_mul
from .vad1 import predict_mul
from .vad import predict_speech
#import aiofiles
import pandas as pd
from fastapi import FastAPI, UploadFile

from predict import load_model

@api_view(['GET'])
def index(request):
    return HttpResponse('Recording Page')

@api_view(['GET'])
def audio(request):
    return response.Response({
        'id':"test"
    })
    return "here"
    print("here")

def get_melspectrogram_db(file_path, sr=None, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80):

  #sr = 16000
  #wav = file_path
  wav,sr = librosa.load(file_path,sr=sr)
  if wav.shape[0]<5*sr:
    wav=np.pad(wav,int(np.ceil((5*sr-wav.shape[0])/2)),mode='reflect')
  else:
    wav=wav[:5*sr]
  spec=librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft,
              hop_length=hop_length,n_mels=n_mels,fmin=fmin,fmax=fmax)
  spec_db=librosa.power_to_db(spec,top_db=top_db)
  return spec_db

def spec_to_image(spec, eps=1e-6):
  mean = spec.mean()
  std = spec.std()
  spec_norm = (spec - mean) / (std + eps)
  spec_min, spec_max = spec_norm.min(), spec_norm.max()
  spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
  spec_scaled = spec_scaled.astype(np.uint8)
  return spec_scaled

def take_log(mels):
    amp_to_db = AmplitudeToDB(stype="amplitude")
    amp_to_db.amin = 1e-5  # amin= 1e-5 as in librosa
    return amp_to_db(mels).clamp(min=-50, max=80)  # clamp to reproduce old code

def scaler(config, log):

    if config["scaler"]["statistic"] == "instance":
        scaler = TorchScaler(
            "instance",
            config["scaler"]["normtype"],
            config["scaler"]["dims"],
        )

        return scaler
    
    if config["scaler"]["savepath"] is not None:
        if os.path.exists(config["scaler"]["savepath"]):
            scaler = torch.load(config["scaler"]["savepath"])
            print(
                "Loaded Scaler from previous checkpoint from {}".format(
                    config["scaler"]["savepath"]
                )
            )
            return scaler

def detect(mel_feats, model, config):
    return model(scaler(config, take_log(mel_feats)))


@api_view(['GET', 'POST'])
def rctVAD(request):
    """
    content= np.array(request.data["data"], np.int16)
    import conv_vad

    vad = conv_vad.VAD()

    # Audio frame is numpy array of 1 sec, 16k, single channel audio data.
    score = vad.score_speech(content)
    if score >=0.5:
        print("yes")
    else:
        print("no")
    response = JsonResponse({'status': 'fail', 'description': 'no audio data detected!!'}, status=status.HTTP_204_NO_CONTENT)
    return response
    """
    OUTPUT_DIR = "chunks/"
    FULL_AUDIO = "final/"

    #app = FastAPI()

    model = load_model()
    
    filepath = './recordings_audio'
    filename = filepath + '/' + str(datetime.now()).replace(' ', '_').replace(':', '_') + '.wav'
    content= np.array(request.data["data"], np.int16)

    wavfile.write(filename, 16000, content)

    #file_path = os.path.join(OUTPUT_DIR, file.filename)
    #with open(file_path, "wb") as f:
    #    f.write(file.file.read())

    prediction: pd.DataFrame = model.predict(filename)
    os.remove(filename)

    total_speech_time = 0
    for index, row in prediction.iterrows():
        total_speech_time += row["offset"] - row["onset"]
        print(total_speech_time)

    if total_speech_time >= 0.8:
        speech_detection = "yes"
    else:
        speech_detection = "no"
    print(speech_detection)
    cheating_level = "no"
    response = JsonResponse({'status': 'success', 'prediction': '', 'speech_detection': speech_detection, 'cheating_level': cheating_level}, status=status.HTTP_200_OK)
    return response

@api_view(['GET', 'POST'])
def speakerrec1(request):
    response = JsonResponse({'status': 'fail', 'description': 'no audio data detected!!'}, status=status.HTTP_204_NO_CONTENT)
    if request.method == 'POST':
        RATE = 16000
        #filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/recordings_audio/"
        #filepath = os.path.join("./recordings_audio/")
        #filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/recordings_audio"
        filepath = './speaker_audio'
        if os.path.exists(filepath) == False:
            os.makedirs(filepath)
        #filename = filepath + "/"+ str(datetime.now()).replace(' ', '_') + '.wav'
        filename = filepath + '/' + str(datetime.now()).replace(' ', '_').replace(':', '_') + '.wav'
        speaker_audio = filepath + '/' + 'speaker2.wav'
        cheating_level = "no"
        #filename = 
        speaker_reco = "yes"

        try:
            #with open(filename, 'wb') as file:
             #   print("file")
            #print(json.loads(request.data))
            #content = np.array(json.loads(request.data), np.int16)
            content= np.array(request.data["data"], np.int16)
            model = torch.load('models/models/silero_vad.jit')
            speech_timestamps = get_speech_timestamps(content, model, sampling_rate=RATE)
            if (len(speech_timestamps) > 0):

                #sf.write(filename, content, 16000)
                #wavfile.write("./recordings_audio/"+'test.wav', 16000, content)
                wavfile.write(filename, 16000, content)
                #wavfile.write("./recordings_audio/"+'2022-07-10_23_21_25.893481.wav', 16000, content)

                verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")

                #score, prediction = verification.verify_files("123.wav", "datasets/SI1265_FJWB0_2.wav") # Different Speakers
                #score, prediction = verification.verify_files("tests/samples/ASR/spk1_snt1.wav", "tests/samples/ASR/spk1_snt2.wav") # Same Speaker
                score, prediction = verification.verify_files(speaker_audio, filename)
                check = prediction.numpy()[0]

                if check == True:
                    speaker_reco = "yes"
                else:
                    speaker_reco = "no"
                print(speaker_reco)
                
                response = JsonResponse({'status': 'success', 'prediction': '', 'speaker_reco': speaker_reco}, status=status.HTTP_200_OK)
                os.remove(filename)
                return response
            else:
                response = JsonResponse({'status': 'success', 'prediction': '', 'speaker_reco': speaker_reco}, status=status.HTTP_200_OK)
                return response
        except:
            return response


"""
@app.post("/full_audio")
def full_audio(file: UploadFile):
    file_path = os.path.join(FULL_AUDIO, file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())
"""

"""
@api_view(['GET', 'POSt'])
def rctVAD(request):
    filepath = './recordings_audio'
    filename = filepath + '/' + str(datetime.now()).replace(' ', '_').replace(':', '_') + '.wav'
    content= np.array(request.data["data"], np.int16)
    wavfile.write(filename, 16000, content)

    MODEL_BASE_DIR = "./result"
    # MODEL_PATH = os.path.join(MODEL_BASE_DIR, "epoch=197-step=20591.ckpt")  # Speech and 9 other classes (10 classes)
    MODEL_PATH = os.path.join(MODEL_BASE_DIR, "epoch=177-step=6407.ckpt")  # Speech only

    # Load config from hparams.yaml
    CONFIG_FILE_PATH = os.path.join(MODEL_BASE_DIR, "hparams.yaml")
    with open(CONFIG_FILE_PATH, "r") as f:
        config = yaml.safe_load(f)
    

    labels = [
        "Alarm_bell_ringing",
        "Blender",
        "Cat",
        "Dishes",
        "Dog",
        "Electric_shaver_toothbrush",
        "Frying",
        "Running_water",
        "Speech",
        "Vacuum_cleaner",
    ]

    feat_params = config["feats"]
    mel_spec = MelSpectrogram(
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
        )


    audio, _ = read_audio(
        filename,
        multisrc=False,
        random_channel=False,  # pad_to=2 * 16000
    )
    audio = torch.tensor(nr.reduce_noise(y=audio, sr=16000))
    audio = torch.reshape(
        audio.to(device="cuda" if torch.cuda.is_available() else "cpu"),
        (1, audio.shape[0]),
    )

    mels = mel_spec(audio)
    sed_student = torch.nn.DataParallel(
        CRNN(**config["net"]).to(device="cuda" if torch.cuda.is_available() else "cpu")
        # CRNN(**config["net"]).to(device="cpu")

    )
    strong_preds, _ = detect(mels, sed_student, config)
    encoder = ManyHotEncoder(
        labels=labels,
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["n_filters"],
        frame_hop=config["feats"]["hop_length"],
        net_pooling=config["data"]["net_subsample"],
        fs=config["data"]["fs"],
    )

    decoded_strong = batched_decode_preds(
        strong_preds,
        [filename],
        encoder,
        median_filter=1,
        thresholds=[0.8],
    )
    return decoded_strong[0.8][decoded_strong[0.8]["event_label"] == "Speech"]

"""  
""" 
@api_view(['GET', 'POST'])
def speechVAD(request):
    response = JsonResponse({'status': 'fail', 'description': 'no audio data detected!!'}, status=status.HTTP_204_NO_CONTENT)
    if request.method == 'POST':
        #filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/recordings_audio/"

        #filepath = os.path.join("./recordings_audio/")
        #filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/recordings_audio"
        filepath = './recordings_audio'
        if os.path.exists(filepath) == False:
            os.makedirs(filepath)
        #filename = filepath + "/"+ str(datetime.now()).replace(' ', '_') + '.wav'
        filename = filepath + '/' + str(datetime.now()).replace(' ', '_').replace(':', '_') + '.wav'
        cheating_level = "no"
        #filename = 
        try:
            #with open(filename, 'wb') as file:
             #   print("file")
            #print(json.loads(request.data))
            #content = np.array(json.loads(request.data), np.int16)
            content= np.array(request.data["data"], np.int16)
            #sf.write(filename, content, 16000)
            #wavfile.write("./recordings_audio/"+'test.wav', 16000, content)
            wavfile.write(filename, 16000, content)
            #wavfile.write("./recordings_audio/"+'2022-07-10_23_21_25.893481.wav', 16000, content)
            cheating_level = env_sound(filename)
        except:
            return response

        val = predict_speech(request, filename)
        if val: 
            if val == "yes":
                cheating_level = "yes"
            response = JsonResponse({'status': 'success', 'prediction': '', 'speech_detection': val, 'cheating_level': cheating_level}, status=status.HTTP_200_OK)
            return response
        return val
"""
@api_view(['GET', 'POST'])
def speechVAD(request):
    from speechbrain.pretrained import VAD
    filepath = './recordings_audio'
    filename = filepath + '/' + str(datetime.now()).replace(' ', '_').replace(':', '_') + '.wav'
    content= np.array(request.data["data"], np.int16)
    wavfile.write(filename, 16000, content)
    VAD = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="pretrained_models/vad-crdnn-libriparty")
    #boundaries = VAD.get_speech_segments("speechbrain/vad-crdnn-libriparty/example_vad.wav")
    boundaries = VAD.get_speech_segments(filename)
    #print(len(boundaries))
    #print(VAD.save_boundaries(boundaries))
    cheating_level = env_sound(filename)
    if len(boundaries) == 0:
        speech_detection = "no"
    else:
        speech_detection = "yes"
        cheating_level = "high"
    #print(speech_detection)
    # Print the output
    #print(VAD.save_boundaries(boundaries))\
    #cheating_level = "low"
    
    response = JsonResponse({'status': 'success', 'prediction': '', 'speech_detection': speech_detection, 'cheating_level': cheating_level}, status=status.HTTP_200_OK)
    return response

def env_sound(filename):
    
    low = ['dog', 'rooster', 'pig', 'cow', 'frog', 'cat', 'hen', 'insects', 'sheep', 'crow', 'rain', 'sea_waves',
'crackling_fire', 'crickets', 'chirping_birds', 'water_drops', 'wind', 'thunderstorm', 'crying_baby', 'sneezing', 'breathing'
'coughing', 'brushing_teeth', 'snoring', 'drinking_sipping', 'mouse_click', 'keyboard_typing', 'can_opening', 'washing_machine',
'vacuum_cleaner', 'clock_alarm', 'clock_tick', 'glass_breaking', 'helicopter', 'chain_saw', 'siren', 'car_horn', 'engine', 'train',
'church_bells', 'airplane', 'crackers', 'hand_saw', 'drilling', 'engine_idling', 'gun_shot', ' jackhammer', 'siren', 'street_music',
'children_playing', 'air_conditioner']

    medium = ['pouring_water', 'toilet_flush', 'laughing']

    high = ['footsteps', 'door_knock', 'door_wood_creaks']

    with open('models/env_model/indtocat.pkl','rb') as f:
        env_model = pickle.load(f)
    with open('models/env_model/esc50resnet.pth','rb') as f:
        resnet_model = torch.load(f, map_location=torch.device('cpu'))
    spec=spec_to_image(get_melspectrogram_db(filename))
    spec_t=torch.tensor(spec).to(dtype=torch.float32)

    pr=resnet_model.forward(spec_t.reshape(1,1,*spec_t.shape))
    ind = pr.argmax(dim=1).cpu().detach().numpy().ravel()[0]
    #print(env_model[ind])
    if env_model[ind] in low:
        return "low"
    elif env_model[ind] in medium:
        return "medium"
    else:
        return "high"

@api_view(['GET', 'POST'])
def sileroVAD(request):

    #env_model = load_model("models/esc50_.46_0.7929_0.8050.hdf5")
    mark = time()
    CHUNK = 1024
    #FORMAT = pyaudio.paInt32
    CHANNELS = 1
    RATE = 16000 
    RECORD_SECONDS = 3
    WAVE_OUTPUT_FILENAME = "test.wav"
    NFRAMES = int((RATE * RECORD_SECONDS) / CHUNK)
    #content = np.array(json.loads(request.data), np.int16)
    filepath = './recordings_audio'
    filename = filepath + '/' + str(datetime.now()).replace(' ', '_').replace(':', '_') + '.wav'
    content= np.array(request.data["data"], np.int16)
    wavfile.write(filename, 16000, content)
    model = torch.load('models/models/silero_vad.jit')
    cheating_level = env_sound(filename)

    speech_timestamps = get_speech_timestamps(content, model, sampling_rate=RATE)
    if (len(speech_timestamps) > 0):
        response = JsonResponse({'status': 'success', 'prediction': '', 'speech_detection': 'yes', 'cheating_level': "high"}, status=status.HTTP_200_OK)
    else:
        response = JsonResponse({'status': 'success', 'prediction': '', 'speech_detection': 'no', 'cheating_level': cheating_level}, status=status.HTTP_200_OK)
    return response

@api_view(['GET', 'POSt'])
def mulspeaker1(request):
    response = JsonResponse({'status': 'fail', 'description': 'no audio data detected!!'}, status=status.HTTP_204_NO_CONTENT)
    if request.method == 'POST':
        #filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/recordings_audio/"

        #filepath = os.path.join("./recordings_audio/")
        #filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/recordings_audio"
        filepath = './recordings_audio'
        if os.path.exists(filepath) == False:
            os.makedirs(filepath)
        #filename = filepath + "/"+ str(datetime.now()).replace(' ', '_') + '.wav'
        filename = filepath + '/' + str(datetime.now()).replace(' ', '_').replace(':', '_') + '.wav'
        #filename = 
        try:
            #with open(filename, 'wb') as file:
             #   print("file")
            #print(json.loads(request.data))
            #content = np.array(json.loads(request.data), np.int16)
            content= np.array(request.data["data"], np.int16)
            #sf.write(filename, content, 16000)
            #wavfile.write("./recordings_audio/"+'test.wav', 16000, content)
            wavfile.write(filename, 16000, content)
            #wavfile.write("./recordings_audio/"+'2022-07-10_23_21_25.893481.wav', 16000, content)
        except:
            return response
        val = predict_mul(request, filename)
        print(val)
        return val
"""


@api_view(['GET','POST'])
def arrayVAD(request):
    response = JsonResponse({'status': 'fail', 'description': 'no audio data detected!!'}, status=status.HTTP_204_NO_CONTENT)
    if request.method == 'POST':
        #filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/recordings_audio/"

        #filepath = os.path.join("./recordings_audio/")
        #filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/recordings_audio"
        filepath = './recordings_audio'
        if os.path.exists(filepath) == False:
            os.makedirs(filepath)
        #filename = filepath + "/"+ str(datetime.now()).replace(' ', '_') + '.wav'
        filename = filepath + '/' + str(datetime.now()).replace(' ', '_').replace(':', '_') + '.wav'
        #filename = 
        try:
            #with open(filename, 'wb') as file:
             #   print("file")
            #print(json.loads(request.data))
            #content = np.array(json.loads(request.data), np.int16)
            content= np.array(request.data["data"], np.int16)
            #sf.write(filename, content, 16000)
            #wavfile.write("./recordings_audio/"+'test.wav', 16000, content)
            wavfile.write(filename, 16000, content)
            #wavfile.write("./recordings_audio/"+'2022-07-10_23_21_25.893481.wav', 16000, content)
        except:
            return response
        val = predict(request, filename)
        print(val)
        return val
"""

@api_view(['GET', 'POST'])
def speakerSample(request):
    response = JsonResponse({'status': 'fail', 'description': 'no audio data detected!!'}, status=status.HTTP_204_NO_CONTENT)
    if request.method == 'POST':
        #filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/recordings_audio/"

        #filepath = os.path.join("./recordings_audio/")
        #filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/recordings_audio"
        filepath = './speaker_audio'
        if os.path.exists(filepath) == False:
            os.makedirs(filepath)
        #filename = filepath + "/"+ str(datetime.now()).replace(' ', '_') + '.wav'
        filename = filepath + '/' + 'speaker.wav'
        #filename = 
        try:
            print(request.files)
            #with open(filename, 'wb') as file:
             #   print("file")
            #print(json.loads(request.data))
            #content = np.array(json.loads(request.data), np.int16)
            print(request.data)
            print(request.data["data"])
            content= np.array(request.data, np.int16)
            #sf.write(filename, content, 16000)
            #wavfile.write("./recordings_audio/"+'test.wav', 16000, content)
            wavfile.write(filename, 16000, content)
            #wavfile.write("./recordings_audio/"+'2022-07-10_23_21_25.893481.wav', 16000, content)
        except:
            return response