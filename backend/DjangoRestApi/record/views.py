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
from silero.utils_vad import get_speech_timestamps, read_audio, VADIterator, get_number_ts, save_audio
from time import time
import scipy.signal as sps
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
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
from rct.dataio.datasets import read_audio as read_audio1
from rct.local.utils import batched_decode_preds
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
import noisereduce as nr
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from record.models import Recording
from record.serializers import RecordingSerializer
import scipy.io.wavfile
from rct.nnet.CRNN import CRNN
import os
from speechbrain.pretrained import SpeakerRecognition
from .vad1 import predict_mul
from .vad import predict_speech
import pandas as pd
from fastapi import FastAPI, UploadFile
from rct.predict import load_model
import wave
from rct.utils.encoder import ManyHotEncoder
from django.core.files.storage import FileSystemStorage
from scipy.io import wavfile
import wave, os, glob
import librosa
import soundfile as sf
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import subprocess
from pydub import AudioSegment
from collections import defaultdict
import time
import sklearn
import pydub
import gtts
from playsound import playsound
import subprocess
from exams.models import Exam

model = torch.load('models/models/silero_vad.jit')
@api_view(['GET'])
def index(request):
    return HttpResponse('Recording Page')

@api_view(['GET'])
def audio(request):
    return response.Response({
        'id':"test"
    })

@api_view(['GET', 'POST'])
def test(request):
   
    if request.method =="POST":
        response = JsonResponse({'status': 'success', 'result':'test'}, status=status.HTTP_200_OK)
        return response
    else:
        print("")
def get_melspectrogram_db(file_path, sr=None, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80):
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

def create_folder(filepath):
    print(filepath)
    if os.path.isdir(filepath)  == False:
        os.makedirs(filepath)
    return True

@api_view(['GET', 'POST'])
def saveFile(request):
    response = JsonResponse({'status': 'fail', 'description': 'no audio data detected!!'}, status=status.HTTP_204_NO_CONTENT)
    if request.method == "POST":
        user_id = request.data["user_id"]
        exam_id = request.data["exam_id"]
        print(type(request.FILES["file"]))
        CHUNK = 1024
        CHANNELS = 1
        RATE = 16000 
        RECORD_SECONDS = 3
        NFRAMES = int((RATE * RECORD_SECONDS) / CHUNK)
        RATE = 16000 
        SAMPLEWIDTH = 2
        file = request.FILES.get("file")
        f = request.FILES["file"]
        import io
        print(file)
        folder_name = settings.MEDIA_URL+"user_audio/exam_"+ str(exam_id) + "/user_" + str(user_id)

        filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + folder_name
        print(filepath)
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        
        filename = str(datetime.now()).replace(' ', '_').replace(':', '_') + "_" + str(exam_id) + "_" + str(user_id)
        f_name = filename + ".webm"
        wav_filename = filename + ".wav"

        with open(filepath + "/" + f_name,  mode='bx') as destination:
            for chunk in f.chunks():
                destination.write(chunk)
            destination.close()

        subprocess.call('ffmpeg -i "'+filepath + '/' + f_name +'" -vn "'+ filepath + '/' + wav_filename+ '"', shell = True)
        basename = os.path.splitext(folder_name + "/" + wav_filename)[0]
        store = Recording(filename = folder_name + "/" + wav_filename, folder_name = basename, created_at = datetime.now(), user_id_id = user_id, exam_id_id = exam_id)
        store.save()
        os.remove(filepath + "/" + f_name)
        exams = Exam.objects.filter(id__exact=exam_id).update(analyze = 0)

        response = JsonResponse({'status': 'success'}, status=status.HTTP_200_OK)
       
    return response

def convert(time_val):
    t = float(time_val)
    hours = int(t // 3600)
    minutes = int((t % 3600) // 60)
    seconds = int(t % 60)
    return hours, minutes, seconds

@api_view(['GET', 'POST'])
def recordingView(request, record_id):
    """
    recording  function fetches the list of record from database
    :param request: contains request data sent from frontend
    param record_id: contains record id sent from frontend
    :return: list of recording
    """ 
    response = JsonResponse({'status': 'fail', 'description': 'no audio data detected!!'}, status=status.HTTP_204_NO_CONTENT)
    if request.method == "POST":
        user_id = request.data['user_id']
        try:
            filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))       
            recordings = Recording.objects.filter(id__exact=record_id)   
            recording_serializer = RecordingSerializer(recordings, many=True)
            recording_data = json.loads(json.dumps(recording_serializer.data))
            org_filename = recording_data[0]["filename"]
            folder_name = settings.MEDIA_URL+"user_audio/"+ str(user_id) + "/" + recording_data[0]["folder_name"]
            text_folder_name= settings.MEDIA_URL+"user_audio/"+ str(user_id) + "/txt_file"
            print(settings.MEDIA_URL)
            print(org_filename)
            data = []
            index = 0            
            text_file = text_folder_name + "/" + recording_data[0]["folder_name"] + ".txt"
            time_detail = []
            with open(filepath + "/" + text_file) as f:                
                lines = f.readlines()[1:]
                for i in range(0, len(lines)):
                    text_list = defaultdict(list)
                    x = lines[i].split(",")
                    hours, minutes, seconds = convert(x[0])
                    print("{}:{}:{}".format(hours, minutes, seconds))
                    text_list["start"].append(str("{}:{}:{}".format(hours, minutes, seconds)))
                    hours, minutes, seconds = convert(x[1])
                    text_list["end"].append(str("{}:{}:{}".format(hours, minutes, seconds)).replace("\n",""))
                    time_detail.append(text_list)

            index = 0
            for filename in os.listdir(filepath +  folder_name):
                file_list = defaultdict(list)
                file_list["split"].append(folder_name + "/" + os.path.splitext(filename)[0])
                file_list["full"].append(folder_name + "/" + filename)
                file_list["start"].append(time_detail[index]["start"][0])
                file_list["end"].append(time_detail[index]["end"][0])   
                data.append(file_list)
                index += 1
            text_file = text_folder_name + "/" + recording_data[0]["folder_name"] + ".txt"
            text_list = defaultdict(list)
            print(org_filename)
            return JsonResponse({"data" : data, "filename" : org_filename}, safe=False)
        except: 
            return JsonResponse({'data': 'fail'}, status=status.HTTP_204_NO_CONTENT)

@api_view(['GET', 'POST'])
def recordingList(request):
    """
    recordinglist  function fetches the list of record from database
    :param request: contains request data sent from frontend
    :return: list of recording
    """ 
    response = JsonResponse({'status': 'fail', 'description': 'no audio data detected!!'}, status=status.HTTP_204_NO_CONTENT)
    if request.method == "POST":
        user_id = request.data['user_id']
        try:
            recordings = Recording.objects.filter(user_id_id__exact=user_id)   
            recording_serializer = RecordingSerializer(recordings, many=True)
            return JsonResponse({"data":recording_serializer.data,"status":"success"}, safe=False)
        except: 
            return JsonResponse({'data': 'fail'}, status=status.HTTP_204_NO_CONTENT)

@api_view(['GET', 'POST'])
def speechCheck(request):
    """
    speechcheck function check the speech from database
    param request: contains request data sent from frontend
    return: speech details
    """
    response = JsonResponse({'status': 'fail', 'description': 'no audio data detected!!'}, status=status.HTTP_204_NO_CONTENT)
    if request.method == "POST":
        user_id = request.data['user_id']
        zero = []        
        folder_name = settings.MEDIA_URL+"user_audio/"+ str(user_id)
        filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + folder_name
        if os.path.exists(filepath):
            filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + folder_name
            print(filepath)
            RATE = 16000 
            print("here")
            for filename in os.listdir(filepath):
                if filename.endswith('.wav'):
                    model = torch.load('models/models/silero_vad.jit')
                    model.reset_states()
                    audio_path = filepath + "/" + filename                    
                    text_file_name = "txt_file" + "/" + os.path.splitext(filename)[0]
                    folder_name = os.path.splitext(filename)[0]
                    
                    if os.path.exists(filepath + "/" + text_file_name + ".txt"):
                        with open(filepath + "/" + text_file_name + ".txt") as f:
                            lines = f.readlines()[1:]
                        for i in lines:
                            x = i.split(",")
                            print(x[1])
                            t1 = int(float(x[0])) * 1000 #Works in milliseconds
                            t2 = int(float(x[1].replace("\n",""))) * 1000
                            print(audio_path)
                            newAudio = AudioSegment.from_wav(audio_path)
                            newAudio = newAudio[t1:t2]
                            basename = os.path.basename(audio_path)
                            filename = os.path.splitext(basename)
                            name = filename[0] + "_" + str(x[0]) + "_" + str(x[1].replace("\n",""))  + filename[1]
                            f_name = filename[0] + "_" + str(x[0]) + "_" + str(x[1].replace("\n",""))
                            if not os.path.exists(filepath + "/" + folder_name):
                                os.mkdir(filepath + "/" + folder_name)
                            new_file_path = filepath + "/" + folder_name + "/" + name
                            newAudio.export(new_file_path , format="wav")
                        
                    else:
                        wav = read_audio(audio_path, RATE)
                        vad_iterator = VADIterator(model)
                        window_size_samples = 1536 # number of samples in a single audio chunk       
                        start = []
                        end = []
                        timestamp = []
                        for i in range(0, len(wav), window_size_samples):
                            chunk = wav[i: i+ window_size_samples]
                            if len(chunk) < window_size_samples:
                                break
                            speech_dict = vad_iterator(chunk, return_seconds=True)
                            if speech_dict:
                                for key, value in speech_dict.items():
                                    if key == "start":
                                        start.append(value)
                                    else:
                                        end.append(value)
                                
                        
                        text_file_name = os.path.splitext(filename)[0]
                        f = open(filepath + "/" + text_file_name + ".txt", "a")
                        for i in range(0, len(start)):
                            timestamp.append("{},{}".format(start[i], end[i]))
                        print('\n'.join(timestamp))
                        f.write("start, end (in seconds)\n")
                        f.write('\n'.join(timestamp))
                        f.close()

                        vad_iterator.reset_states() 
        else:
            print("no")
            response = JsonResponse({'status': 'fail'}, status=status.HTTP_200_OK)
            return response
           
            if (len(speech_timestamps) > 0):
                print(speech_timestamps)
                response = JsonResponse({'status': 'success', 'prediction': '', 'speech_detection': 'yes', 'cheating_level': "high"}, status=status.HTTP_200_OK)
            else:
                response = JsonResponse({'status': 'success', 'prediction': '', 'speech_detection': 'no', 'cheating_level': cheating_level}, status=status.HTTP_200_OK)

@api_view(['GET', 'POST'])
def noisedetection(request):
    """
    noisedetection function detect the noise
    param request: contains request data sent from frontend
    return: detection noise details
    """ 
    noise_presence = "no"
    response = JsonResponse({'status': 'fail', 'description': 'no audio data detected!!'}, status=status.HTTP_204_NO_CONTENT)
    if request.method == 'POST':
        RATE = 16000
        SAMPLE_RATE = 16000
        filepath = './recordings_audio'
        if os.path.exists(filepath) == False:
            os.makedirs(filepath)
        filename = filepath + '/' + str(datetime.now()).replace(' ', '_').replace(':', '_') + '.wav'
        content= np.array(request.data["data"], np.int16)
        wavfile.write(filename, 16000, content)

        try:
           
            response = JsonResponse({'status': 'success', 'prediction': '', 'noise_presence': noise_presence}, status=status.HTTP_200_OK)
            return response
        except:
            response = JsonResponse({'status': 'success', 'prediction': '', 'noise_presence': noise_presence}, status=status.HTTP_200_OK)
            return response
    else:
        response = JsonResponse({'status': 'success', 'prediction': '', 'noise_presence': noise_presence}, status=status.HTTP_200_OK)
        return response


@api_view(['GET', 'POST'])
def speakerrec(request):
    """
    speakerrec function recognition of speech
    param request: contains request data sent from frontend
    return: speech reconginition details
    """ 
    response = JsonResponse({'status': 'fail', 'description': 'no audio data detected!!'}, status=status.HTTP_204_NO_CONTENT)
    if request.method == 'POST':
        user_id = request.data["user_id"]
        RATE = 16000
        folder_name = settings.MEDIA_URL+"speaker_audio"   
        filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + folder_name
        speaker_audio = filepath + '/speaker_sample_' + str(user_id) + '.wav' 
        filename = filepath + '/' + str(datetime.now()).replace(' ', '_').replace(':', '_') + '.wav'
        speaker_reco = "yes"

        try: 
            content= np.array(request.data["data"], np.int16)
            model = torch.load('models/models/silero_vad.jit')
            speech_timestamps = get_speech_timestamps(content, model, sampling_rate=RATE)
            if (len(speech_timestamps) > 0):  
                wavfile.write(filename, 16000, content)
                verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
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

@api_view(['GET', 'POSt'])
def rctVAD(request):
    filepath = './recordings_audio'
    filename = filepath + '/' + str(datetime.now()).replace(' ', '_').replace(':', '_') + '.wav'
    content= np.array(request.data["data"], np.int16)
    wavfile.write(filename, 16000, content)
    MODEL_BASE_DIR = "./result"
    MODEL_PATH = os.path.join(MODEL_BASE_DIR, "epoch=177-step=6407.ckpt")  
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
    audio, _ = read_audio1(
        filename,
        multisrc=False,
        random_channel=False,  
    )
    audio = torch.tensor(nr.reduce_noise(y=audio, sr=16000))
    audio = torch.reshape(
        audio.to(device="cuda" if torch.cuda.is_available() else "cpu"),
        (1, audio.shape[0]),
    )
    mels = mel_spec(audio)
    sed_student = torch.nn.DataParallel(
        CRNN(**config["net"]).to(device="cuda" if torch.cuda.is_available() else "cpu")
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

@api_view(['GET', 'POST'])
def speechVAD(request):
    response = JsonResponse({'status': 'fail', 'description': 'no audio data detected!!'}, status=status.HTTP_204_NO_CONTENT)
    if request.method == "POST":
        user_id = request.data["user_id"]
        try:
            from speechbrain.pretrained import VAD
            filepath = './uploads/recordings_audio'
            filename = filepath + '/' + str(datetime.now()).replace(' ', '_').replace(':', '_') + "_" + str(user_id) +'.wav'
            content= np.array(request.data["data"], np.int16)
            wavfile.write(filename, 16000, content)
            VAD = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="pretrained_models/vad-crdnn-libriparty")  
            boundaries = VAD.get_speech_segments(filename)
            
            cheating_level = env_sound(filename)
            if len(boundaries) == 0:
                speech_detection = "no"
            else:
                speech_detection = "yes"
                cheating_level = "high"
           
            
            response = JsonResponse({'status': 'success', 'prediction': '', 'speech_detection': speech_detection, 'cheating_level': cheating_level}, status=status.HTTP_200_OK)
            return response
        except:
            return response
    elif(request == "GET"):
        print("what")
    else:
        return response

def env_sound(filename):
    """
    env_sound function fetech the enviroment sound
    param filename: contains filename data sent from frontend
    return: sounds from the enviroment
    """     
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
    if env_model[ind] in low:
        return "low"
    elif env_model[ind] in medium:
        return "medium"
    else:
        return "high"

@api_view(['GET', 'POST'])
def sileroVAD(request): 
    response = JsonResponse({'status': 'fail', 'description': 'no audio data detected!!'}, status=status.HTTP_204_NO_CONTENT)
    if request.method == "POST":
        user_id = request.data["user_id"]
        
        try:
            CHUNK = 1024
            CHANNELS = 1
            RATE = 16000 
            RECORD_SECONDS = 3
            NFRAMES = int((RATE * RECORD_SECONDS) / CHUNK)
            cheating_level = "low"
            model = torch.load('models/models/silero_vad.jit')
            filepath = './uploads/recordings_audio'
            filename = filepath + '/' + str(datetime.now()).replace(' ', '_').replace(':', '_') + "_" + str(user_id) +'.wav'
            content= np.array(request.data["data"], np.int16)
            wavfile.write(filename, 16000, content)
            model.reset_states() 
            cheating_level = env_sound(filename)
            speech_timestamps = get_speech_timestamps(content, model, sampling_rate=RATE)
            if (len(speech_timestamps) > 0):
                response = JsonResponse({'status': 'success', 'prediction': '', 'speech_detection': 'yes', 'cheating_level': "high"}, status=status.HTTP_200_OK)
            else:
                response = JsonResponse({'status': 'success', 'prediction': '', 'speech_detection': 'no', 'cheating_level': cheating_level}, status=status.HTTP_200_OK)
            os.remove(filename)           
            return response
        except:
            return response
    else:
        return response

def count(audio, model, scaler):
    X = np.abs(librosa.stft(audio, n_fft=400, hop_length=160)).T  
    X = scaler.transform(X)
    X = X[:500, :]
    Theta = np.linalg.norm(X, axis=1) + eps
    X /= np.mean(Theta)
    X = X[np.newaxis, ...]
    if len(model.input_shape) == 4:
        X = X[:, np.newaxis, ...]
    ys = model.predict(X, verbose=0)
    return np.argmax(ys, axis=1)[0]

@api_view(['GET', 'POSt'])
def mulspeaker1(request):
    """
    mulspeaker function fetech the multispeaker
    param request: contains request data sent from frontend
    return: multiplespeakers details 
    """ 
    response = JsonResponse({'status': 'fail', 'description': 'no audio data detected!!'}, status=status.HTTP_204_NO_CONTENT)
    if request.method == 'POST':
        filepath = './uploads/recordings_audio'
        if os.path.exists(filepath) == False:
            os.makedirs(filepath)
        filename = filepath + '/' + str(datetime.now()).replace(' ', '_').replace(':', '_') + '.wav'
        try:          
            print("model")
            model = torch.load('models/env_classification.h5')           
            print(model)
            model_path = os.path.join('/models/multi_speaker_model', 'CRNN.h5')
            model = keras.models.load_model(model_path)
            print(model)           
            model.summary()
            print(model.summary())           
            scaler = sklearn.preprocessing.StandardScaler()
            with np.load(os.path.join("models", 'scaler.npz')) as data:
                scaler.mean_ = data['arr_0']
                scaler.scale_ = data['arr_1']
                content= np.array(request.data["data"], np.int16)            
                audio = np.mean(content, axis=1)
                estimate = count(audio, model, scaler)
                print("Speaker Count Estimate: ", estimate)         
        except:
            return response
        
@api_view(['GET', 'POST'])
def speakerSample(request):
    """
    speaker_sample function fetched the details of speaker from database
    :param request: contains request data sent from frontend
    :return: return the speakers details
    """
    response = JsonResponse({'status': 'fail', 'description': 'no audio data detected!!'}, status=status.HTTP_204_NO_CONTENT)
    if request.method == 'POST':
        user_id = request.data["user_id"]
        folder_name = settings.MEDIA_URL+"speaker_audio"
        f = request.FILES["file"]   
        filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + folder_name
        if os.path.exists(filepath) == False:
            os.makedirs(filepath)
        filename = filepath + '/speaker_sample_' + str(user_id) + '.wav'
        f_name = 'speaker_sample_' + str(user_id) + ".webm"
        wav_filename = 'speaker_sample_' + str(user_id) + ".wav"
        print(filepath + "/" + f_name)
        try:  
            with open(filepath + "/" + f_name,  mode='bx') as destination:
                for chunk in f.chunks():
                    destination.write(chunk)
                destination.close()
            subprocess.call('ffmpeg -y -i "'+filepath + '/' + f_name +'" -vn "'+ filepath + '/' + wav_filename+ '"') # check the ffmpeg command line :)
            os.remove(filepath + "/" + f_name)
            response = JsonResponse({'status':'success'}, status=status.HTTP_200_OK)
            return response
        except:
            return response