from django.conf.urls import url 
from django.urls import path
# from django.conf.urls import url
from django.urls import path, include
# from django.conf.urls import url
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
    path('api/rctVAD', views.rctVAD, name = "rctVAD")
    #path('api/arrayVAD',  views.arrayVAD, name = "arrayVAD"),
    #url('api/arrayVAD',  views.arrayVAD, name = "arrayVAD"),

]

