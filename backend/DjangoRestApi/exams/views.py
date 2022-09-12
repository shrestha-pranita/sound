from django.http import HttpResponse
from django.shortcuts import render
from rest_framework.decorators import api_view
from exams.models import Exam
#from recordings.models import Recording


from exams.serializers import ExamSerializer, ExamForeignSerializer
#from recordings.serializers import RecordingSerializer, RecordingForeignSerializer

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
from record.models import Recording

@api_view(['GET', 'POST', 'DELETE'])
def exam_list(request):
    #print(request.META['Authorization'])
    #permission_classes = (IsAuthenticated,) 
    #print(request.headers['Authorization'])
    if request.method == 'GET':
        try:
            exams = Exam.objects.filter(status=1)
            exam_serializer = ExamSerializer(exams, many=True)
            return JsonResponse(exam_serializer.data, safe=False)
            #exam_id = Question.objects.filter(status=1).values('exam_id').distinct()
            #exams = Exam.objects.filter(id__in = exam_id).filter(status__exact=1)   
            #print(exams)
            #exam_serializer = ExamSerializer(exams, many=True)
            #return JsonResponse(exam_serializer.data, safe=False)
        except: 
            return JsonResponse({'message': 'There are no exams at the moment'}, status=status.HTTP_404_NOT_FOUND)

@api_view(['GET', 'POST'])
def exam_detail(request, exam_id):
    print("herehehheheheheh")
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

"""
@api_view(['GET', 'POST', 'DELETE'])
#@authentication_classes([SessionAuthentication, BasicAuthentication])
@permission_classes([IsAuthenticated])
def exam_list(request):
    #print(request.META['Authorization'])
    #permission_classes = (IsAuthenticated,) 
    #print(request.headers['Authorization'])
    if request.method == 'GET':
        try:
            print("here")
            exam_id = Question.objects.filter(status=1).values('exam_id').distinct()
            exams = Exam.objects.filter(id__in = exam_id).filter(status__exact=1)   
            print(exams)
            exam_serializer = ExamSerializer(exams, many=True)
            return JsonResponse(exam_serializer.data, safe=False)
        except: 
            return JsonResponse({'message': 'There are no exams at the moment'}, status=status.HTTP_404_NOT_FOUND)
    
    elif request.method == 'POST':
        exam_data = JSONParser().parse(request)
        exam_serializer = ExamSerializer(data=exam_data)
        if exam_serializer.is_valid():
            exam_serializer.save()
            return JsonResponse(exam_serializer.data, status=status.HTTP_201_CREATED) 
        return JsonResponse(exam_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    elif request.method == 'DELETE':
        count = Exam.objects.all().delete()
        return JsonResponse({'message': '{} Exams were deleted successfully!'.format(count[0])}, status=status.HTTP_204_NO_CONTENT)

@api_view(['GET'])
@permission_classes((IsAuthenticated,))
def recording_list(request):
    id = request.user.id
    if request.method == 'GET':
        try:
            recordings = Recording.objects.filter(user_id=id).select_related('user_id')
            #recordings = RecordingForeignSerializer.objects.filter(user_id_id__exact=id) 
            #query = "select recordings.id, recordings.recording_start_time recording.created_at as created_at, question.question_text as question_text, user.username as user from recording inner join question inner join user on recording.question_id = question.id and recording.user_id=user.id"  
            recording_serializer = RecordingForeignSerializer(recordings, many=True)
            return JsonResponse(recording_serializer.data, safe=False)
        except: 
            return JsonResponse({'message': 'There are no exams at the moment'}, status=status.HTTP_404_NOT_FOUND)

"""