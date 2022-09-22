from django.urls import path
from . import views
from django.conf.urls import include, url, re_path
from django.contrib import admin


urlpatterns = [
    path('api/exam', views.exam_list, name = 'exam_list'),
    url('api/exam', views.exam_list, name = 'exam_list'),
    path('api/startexam/<int:exam_id>', views.exam_detail, name = 'exam_detail'),

]

