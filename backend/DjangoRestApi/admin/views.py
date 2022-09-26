from django.http import HttpResponse
from django.shortcuts import render
from rest_framework.decorators import api_view
from exams.models import Exam
from record.models import Recording
from record.serializers import RecordingSerializer, RecordingForeignSerializer
from django.core import serializers
from exams.serializers import ExamSerializer, ExamForeignSerializer
from django.http import JsonResponse
from rest_framework import permissions, status
from rest_framework.parsers import JSONParser 
from rest_framework import viewsets
from rest_framework.response import Response
import json
from django.http import HttpResponse
from django.forms.models import model_to_dict
from django.shortcuts import get_object_or_404
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import SessionAuthentication, BasicAuthentication
from rest_framework.decorators import authentication_classes, permission_classes
import os
from django.conf import settings
import torch
from silero.utils_vad import get_speech_timestamps, read_audio, VADIterator, get_number_ts, save_audio
from pydub import AudioSegment
from collections import defaultdict
import librosa
import io
import soundfile

@api_view(['GET', 'POST', 'DELETE'])
def admin_exam_list(request):
    """
    admin_exam_list function fetches the list of exam from database
    :param request: contains request data sent from frontend
    :return: list of exam details
    """  
    if request.method == 'GET':
        try:
            exams = Exam.objects.filter(status=1)
            print(exams)
            exam_serializer = ExamSerializer(exams, many=True)
            return JsonResponse(exam_serializer.data, safe=False)
        except: 
            return JsonResponse({'message': 'There are no exams at the moment'}, status=status.HTTP_404_NOT_FOUND)


@api_view(['GET', 'POST', 'DELETE'])
def admin_record_list(request, exam_id):
    """
    admin_record_list function fetches the list of record from database
    :param request: contains request data sent from frontend
    param exam_id: contrains exam id sent from frontend
    :return: list of records details
    """  
    response = JsonResponse({'status': 'fail', 'description': 'no audio data detected!!'}, status=status.HTTP_204_NO_CONTENT)
    if request.method == "POST":
        exam_id = request.data["exam_id"]
        try:
            exams = Exam.objects.filter(id__exact=exam_id)
            exam_serializer = ExamSerializer(exams, many = True)

            recordings = Recording.objects.filter(exam_id_id__exact=exam_id) 
            recording_serializer = RecordingForeignSerializer(recordings, many=True) 
            return JsonResponse({"data":recording_serializer.data, "exam": exam_serializer.data, "status":"success"}, safe=False)
        except: 
            return JsonResponse({'data': 'fail'}, status=status.HTTP_204_NO_CONTENT)
    else:
        return response

def dissect_speech(audio_path, RATE, model):
    wav = read_audio(audio_path, RATE)
    vad_iterator = VADIterator(model)
    window_size_samples = 1536    
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
    if len(start) != len(end):
        d = librosa.get_duration(filename=audio_path)
        end.append(d)
    
    vad_iterator.reset_states() 
    model.reset_states()
    print(start)
    return start, end


def convert(time_val):    
    t = float(time_val)
    hours = int(t // 3600)
    minutes = int((t % 3600) // 60)
    seconds = int(t % 60)
    return hours, minutes, seconds

@api_view(['GET', 'POST'])
def recordingViews(request, record_id):
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
            folder_name = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + recording_data[0]["folder_name"]
            text_folder_name = filepath + settings.MEDIA_URL+ "user_audio/exam_"+ str(recording_data[0]["exam_id"]) + "/txt_file"
            audio_folder_name = recording_data[0]["folder_name"] 
            print(audio_folder_name)
            data = []
            index = 0
            text_file_name = os.path.basename(recording_data[0]["folder_name"]) + ".txt"
            print(text_file_name)
            text_file = os.path.basename(recording_data[0]["folder_name"]) + ".txt"
            time_detail = []
            print(filepath + "/" + text_folder_name + "/" + text_file_name)

            with open(text_folder_name + "/" + text_file_name) as f:
                lines = f.readlines()[1:]
                print(lines)
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
            for filename in os.listdir(folder_name):
                file_list = defaultdict(list)
                file_list["split"].append(audio_folder_name + "/" + os.path.splitext(filename)[0])
                file_list["full"].append(audio_folder_name + "/" + filename)
                file_list["start"].append(time_detail[index]["start"][0])
                file_list["end"].append(time_detail[index]["end"][0])
   
                data.append(file_list)
                index += 1

            return JsonResponse({"data" : data, "filename" : org_filename}, safe=False)
        except: 
            return JsonResponse({'data': 'fail'}, status=status.HTTP_204_NO_CONTENT)


@api_view(['GET', 'POST'])
def userRecordingViews(request, record_id):
    """
    userrecording:  function fetches the list of record of users from database
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
            folder_name = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + recording_data[0]["folder_name"]
            text_folder_name = filepath + settings.MEDIA_URL+ "user_audio/exam_"+ str(recording_data[0]["exam_id"]) + "/txt_file"
            audio_folder_name = recording_data[0]["folder_name"] 
            print(audio_folder_name)
            data = []
            index = 0
            text_file_name = os.path.basename(recording_data[0]["folder_name"]) + ".txt"
            print(text_file_name)
            text_file = os.path.basename(recording_data[0]["folder_name"]) + ".txt"
            time_detail = []
            print(filepath + "/" + text_folder_name + "/" + text_file_name)

            with open(text_folder_name + "/" + text_file_name) as f:
                lines = f.readlines()[1:]
                print(lines)
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
            for filename in os.listdir(folder_name):
                file_list = defaultdict(list)        
                file_list["split"].append(audio_folder_name + "/" + os.path.splitext(filename)[0])
                file_list["full"].append(audio_folder_name + "/" + filename)
                file_list["start"].append(time_detail[index]["start"][0])
                file_list["end"].append(time_detail[index]["end"][0])   
                data.append(file_list)
                index += 1

            return JsonResponse({"data" : data, "filename" : org_filename}, safe=False)
        except: 
            return JsonResponse({'data': 'fail'}, status=status.HTTP_204_NO_CONTENT)


@api_view(['GET', 'POST', 'DELETE'])
def admin_analyze(request, exam_id):
    """
    admin_analyze function fetched the admin 
    :param request: contains request data sent from frontend
    :param exam_id: fetching user_id from frontend
    :return: return staus
    """
    response = JsonResponse({'status': 'fail', 'description': 'no audio data detected!!'}, status=status.HTTP_204_NO_CONTENT)
    if request.method == "POST":
        exam_id = request.data["exam_id"]
        try:
            RATE = 16000 
            recordings = Recording.objects.filter(exam_id_id__exact=exam_id)
            recording_serializer = RecordingSerializer(recordings, many=True) 
            filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  

            model = torch.load('models/models/silero_vad.jit')
            model.reset_states()
            exam_text_folder = filepath + settings.MEDIA_URL+ "user_audio/exam_"+ str(exam_id) + "/txt_file"
            if not os.path.isdir(exam_text_folder):
                os.makedirs(exam_text_folder)
           
            for i in recording_serializer.data:
                folder_name = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + i["folder_name"]
                if not os.path.isdir(folder_name):
                    if not os.path.exists(folder_name):
                        os.makedirs(folder_name)

                    filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + i["filename"] 
                    text_file_name = os.path.basename(i["folder_name"]) + ".txt"
                    f = open(exam_text_folder + "/" + text_file_name, "w")
                    start = []
                    end = []
                    timestamp = []

                    wholeAudio = AudioSegment.from_wav(filepath)

                    start, end = dissect_speech(filepath, RATE, model)
                    f = open(exam_text_folder + "/" + text_file_name, "w")
                    f.write("start, end (in seconds)\n")
                    if len(start) !=0:
                        for i in range(0, len(start)):

                            t1 = int(start[i]) * 1000 
                            t2 = int(end[i]) * 1000

                            if t1 != t2:
                                newAudio = wholeAudio[t1:t2]
                                basename = os.path.basename(filepath)
                                filename = os.path.splitext(basename)

                                name = filename[0] + "_" + str(start[i]) + "_" + str(end[i])  + filename[1]
                                f_name = filename[0] + "_" + str(start[i]) + "_" + str(end[i])

                                new_file_path = folder_name + "/" + name
                                newAudio.export(new_file_path , format="wav")

                                timestamp.append("{},{}".format(start[i], end[i]))

                        if len(timestamp) != 0:
                            f.write('\n'.join(timestamp))
                    f.close()
            
            exams = Exam.objects.filter(id__exact=exam_id).update(analyze = 1)
            return JsonResponse({'status': 'success'}, status=status.HTTP_200_OK)
        except: 
            return JsonResponse({'data': 'fail'}, status=status.HTTP_204_NO_CONTENT)

@api_view(['GET', 'POST'])
def exam_detail(request, exam_id):
    """
    exam_details function fetched the details of exam from database
    :param request: contains request data sent from frontend
    :param exam_id: fetching user_id from frontend
    :return: return the message if there is no exam active


    """
    if request.method == 'POST':
        try:
            user_id = request.data['user_id']
            exam_id = request.data['exam_id']
            exams = Exam.objects.filter(id__in = exam_id).filter(status__exact=1)
            print(exams)
            exam_serializer = ExamSerializer(exams, many=True)
            print(exam_serializer)
            return JsonResponse(exam_serializer.data, safe=False)
        except:
            return JsonResponse({'message': 'Exam details not available'})

