from django.conf.urls import url 
from django.urls import path
from users import views 
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
)

from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework_simplejwt.views import TokenObtainPairView

class MyTokenObtainPairSerializer(TokenObtainPairSerializer):
    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)

        # Add custom claims
        token['is_approved'] = user.is_approved
        token['contract_signed'] = user.contract_signed
        # ...
        print(token)
        return token

class MyTokenObtainPairView(TokenObtainPairView):
    serializer_class = MyTokenObtainPairSerializer
 

urlpatterns = [ 
    url('api/users', views.user_list),
    #url('api/register', views.register),
    path('api/register',  views.register, name = "register"),
    path('api/login',  views.login, name = "login"),
    #path('api/logout',  views.logout, name = "logout"),
    #path('api/login/', MyTokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/user/detail', views.get_detail)

    #url(r'^api/tutorials/published$', views.tutorial_list_published)
    #path('',views.login),
    #path('api/login', views.login),
    #path('api/users', views.user_list),
]