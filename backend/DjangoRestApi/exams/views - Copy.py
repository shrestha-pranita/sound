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
        
       # serializer = ExamSerializer(queryset, many=True)
        #return Response(serializer.data)
        return ExamSerializer

    def list(self, request, *args, **kwargs):
        exams = Exam.objects.all()
        serializer = ExamSerializer(exams, many=True)
        oi_dict = model_to_dict(serializer)
        oi_serialized = json.dumps(oi_dict)
        return oi_serialized

    
    """
    def retrieve(self, request, pk=None):
        queryset = Exam.objects.all()
        exams = get_object_or_404(queryset, pk=pk)
        serializer = ExamSerializer(exams)
        return Response(serializer.data)
    """
    #return Response(results.data)

    """
    print("view")
    exams_serializer = ExamSerializer
    def get_queryset(self):
        queryset = Exam.objects.all()
        return queryset

    def list(self, request):
        print("here")
        results = ExamSerializer(self.queryset, many = True)
        return Response(results.data)
    """
    #print(exams_serializer.data)
    #return JsonResponse(exams_serializer.data, safe=False)

    #exams_serializer = ExamSerializer
    #queryset = Exam.objects.all().order_by("exam_name")
    #return JsonResponse(exams_serializer.data, safe=False) 

"""
    print("here")

    def get_queryset(self):
        queryset = Exam.objects.all()
        return queryset

    def list(self, request):
        results = ExamSerializer(self.queryset, many = True)
        return Response(results.data)
   
    #serializer_class = ExamSerializer
    queryset = Exam.objects.all().order_by("exam_name")
    print(queryset)
    serializer_class = ExamSerializer
    #print(exams)

    print("ere")
    def get_queryset(self):
        data = Exam.objects.all()
        print(data)
        for item in data:
            item['exam_name'] = model_to_dict(item['exam_name'])

        return HttpResponse(json.simplejson.dumps(data), mimetype="application/json")
        return queryset
        # filter queryset based on logged in user
        exams = Exam.objects.all().values()
        exam_list = list(exams)
        #exams_serializer = ExamSerializer(exams)
        print("Here")
        print(exam_list)
        #print(exams_serializer)
        data = json.dumps(exam_list)
        
        return HttpResponse(data, content_type="application/json")
        #return JsonResponse(exams_serializer.data, safe=False) 

@api_view(['GET','POST'])
def exam_list(request):
    print("Hrrtr")
    if request.method == 'GET':
        
        try:
            exams = Exam.objects.all().values() 
            exam_list = list(exams)
            return JsonResponse({'data':exam_list})
            print(exam_list)
            # Convert List of Dicts to JSON
            data = json.dumps(exam_list)
            print(data)
            return HttpResponse(data, content_type="application/json")
            exams_serializer = ExamSerializer(exams)
            print(exams_serializer.data)
            return Response(exams_serializer.data)
            #exam_list = list(exams) 
            print(exams_serializer)
            return JsonResponse(['a', 'b'], safe=False) 
            print(exam_list)
            print(JsonResponse(exam_list, safe=False))
            print("test")
            return JsonResponse(exam_list, safe=False)
            print(exams)
            return JsonResponse(exams)
            #return JsonResponse({'messsage':'success'})
            #return JsonResponse(exams_serializer.data, safe=False)
        except: 
            print("nop")
            return JsonResponse({'message': 'The test'}, status=status.HTTP_404_NOT_FOUND)
"""