from django.urls import path
from . import views
from django.conf.urls import include, url, re_path
from django.contrib import admin
#from exams.views import ExamViewSet


urlpatterns = [
    path('api/admin_exam', views.admin_exam_list, name = 'admin_exam_list'),
    path('api/admin_record/<int:exam_id>', views.admin_record_list, name = 'admin_record_list'),
    path('api/admin_analyze/<int:exam_id>', views.admin_analyze, name = 'admin_analyze'),
    path('api/admin_record_view/<int:record_id>', views.recordingViews, name = 'recordingViews'),
    path('api/admin_user_record_view/<int:record_id>',views.userRecordingViews,name='userRecordingViews')
    #url('api/exams', views.exam_list),
    #url('api/recordlist', views.recording_list, name = "recording_list"),
]
                               
#urlpatterns += router.urls