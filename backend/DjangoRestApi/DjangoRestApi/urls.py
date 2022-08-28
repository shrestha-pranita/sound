# from django.conf.urls import url, include
from django.contrib import admin
from django.urls import path, include
from django.conf.urls import url 

urlpatterns = [
    url(r'^', include('record.urls')),
    url('users/',include('users.urls')),
    url('', include('record.urls')),
    #url(r'^', include('user_profile.urls')),
    #url('api', include('user_profile.urls')),
    url(r'^', include('users.urls')),
    #path('questions/', include('questions.urls')),
 
]
