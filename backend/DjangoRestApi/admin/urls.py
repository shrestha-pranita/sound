from django.urls import path
from . import views
from django.conf.urls import include, url, re_path
from django.contrib import admin

urlpatterns = [
    path('api/admin_exam', views.admin_exam_list, name = 'admin_exam_list'),
    path('api/admin_record/<int:exam_id>', views.admin_record_list, name = 'admin_record_list'),
    path('api/admin_analyze/<int:exam_id>', views.admin_analyze, name = 'admin_analyze'),
    path('api/admin_record_view/<int:record_id>', views.recordingViews, name = 'recordingViews'),
    path('api/admin_user_record_view/<int:record_id>',views.userRecordingViews,name='userRecordingViews'),
    path('api/admin_exam_create',views.admin_exam_create, name='admin_exam_create')
]