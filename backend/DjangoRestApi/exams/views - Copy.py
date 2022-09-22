from django.http import HttpResponse
from django.shortcuts import render
from rest_framework.decorators import api_view
from exams.models import Exam
from exams.serializers import ExamSerializer
from django.http import JsonResponse
from rest_framework import permissions, status
from rest_framework.parsers import JSONParser 
from rest_framework import viewsets
from rest_framework.response import Response
import json
from django.http import HttpResponse
from django.forms.models import model_to_dict
from django.shortcuts import get_object_or_404

class ExamViewSet(viewsets.ModelViewSet):
    queryset = Exam.objects.all()

    def get_serializer_class(self):
        
      
        return ExamSerializer

    def list(self, request, *args, **kwargs):                  
        exams = Exam.objects.all()
        serializer = ExamSerializer(exams, many=True)
        oi_dict = model_to_dict(serializer)
        oi_serialized = json.dumps(oi_dict)
        return oi_serialized

    
   