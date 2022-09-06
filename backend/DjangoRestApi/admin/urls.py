from django.urls import path
from . import views
from django.conf.urls import include, url, re_path
from django.contrib import admin
#from exams.views import ExamViewSet


urlpatterns = [
    path('api/admin_exam', views.admin_exam_list, name = 'admin_exam_list'),
    path('api/admin_record/<int:exam_id>', views.admin_record_list, name = 'admin_record_list'),
    #url('api/exams', views.exam_list),
    #url('api/recordlist', views.recording_list, name = "recording_list"),
]

#urlpatterns += router.urls