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
from .vad import predict
import soundfile as sf

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