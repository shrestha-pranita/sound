from django.conf.urls import url 
from django.urls import path
from django.urls import path, include
from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.conf import settings
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.contrib.staticfiles.urls import static
from django.conf.urls import include, url

urlpatterns = [
    path('', views.index, name='index'),
    path('api/audio',  views.audio, name = "record"),
    path('api/sileroVAD',  views.sileroVAD, name = "sileroVAD"),
    path('api/speechVAD',  views.speechVAD, name = "speechVAD"),
    path('api/rctVAD', views.rctVAD, name = "rctVAD"),
    path('api/mulspeaker1', views.mulspeaker1, name="mulspeaker1"),
    path('api/speakerrec1', views.speakerrec1, name="speakerrec1"),
    path('api/speakerSample', views.speakerSample, name="speakerSample"),
    path('api/noisedetection', views.noisedetection, name="noisedetection"),
    path('api/test',  views.test, name = "test"),
    path('api/speech', views.speechCheck, name="speechCheck"),
    path('api/uploads', views.saveFile, name="saveFile"),
]

