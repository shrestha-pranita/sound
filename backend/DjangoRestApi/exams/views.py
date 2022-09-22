from django.http import HttpResponse
from django.shortcuts import render
from rest_framework.decorators import api_view
from exams.models import Exam
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
from record.models import Recording

@api_view(['GET', 'POST', 'DELETE'])
def exam_list(request):

    """
    exam_list function fetches the list of exam from database
    :param request: contains request data sent from frontend
    :return: list of exam details
    """     
    if request.method == 'GET':

        try:
            exams = Exam.objects.filter(status=1) 
            exam_serializer = ExamSerializer(exams, many=True) 
            return JsonResponse(exam_serializer.data, safe=False) 
            
        except: 
            return JsonResponse({'message': 'There are no exams at the moment'}, status=status.HTTP_404_NOT_FOUND)
 

@api_view(['GET', 'POST'])
def exam_detail(request, exam_id):
    """
    exam_details function fetched the 
    :param request: contains request data sent from frontend
    :param exam_id: fetching user_id from frontend
    :return: return the message if there is no exam active


    """
   
    if request.method == 'POST':
      
        try:
            user_id = request.data['user_id']  
            exam_id = request.data['exam_id']   
            exams = Exam.objects.filter(id__in = exam_id).filter(status__exact=1)  
            exam_serializer = ExamSerializer(exams, many=True)  
        
            return JsonResponse(exam_serializer.data, safe=False)  
        except:
            return JsonResponse({'message': 'Exam details not available'})
         
