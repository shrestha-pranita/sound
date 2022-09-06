from django.urls import path
from . import views
from django.conf.urls import include, url, re_path
from django.contrib import admin
#from exams.views import ExamViewSet



"""
urlpatterns = [
    #path('', views.index, name='index'),
    #re_path(r'^api/exams/$', views.exam_list),
    #path('api/exams', views.exam_list, name='exam_list'),
    #url('exams', views.exam_list),
    url('api/exams', views.exam_list),
    url(r'^exams/$', views.exam_list, name='exam_list'),
    #url(r'^api/exams$', views.exam_list),
    #url('exams', views.exam_list),
    #url(r'^api/exams$', views.exam_list),
]
"""
"""
from rest_framework import routers
from . import views
router = routers.DefaultRouter()
router.register('', ExamViewSet, basename = 'Exam')
router.register(r'', ExamViewSet, basename='Exam')
"""

urlpatterns = [
    path('api/exam', views.exam_list, name = 'exam_list'),
    url('api/exam', views.exam_list, name = 'exam_list'),
    path('api/startexam/<int:exam_id>', views.exam_detail, name = 'exam_detail'),
    #url('api/exams', views.exam_list),
    #url('api/recordlist', views.recording_list, name = "recording_list"),
]

#urlpatterns += router.urls