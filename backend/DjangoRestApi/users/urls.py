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
        return token

class MyTokenObtainPairView(TokenObtainPairView):
    serializer_class = MyTokenObtainPairSerializer
urlpatterns = [ 
    url('api/users', views.user_list),
    path('api/register',  views.register, name = "register"),
    path('api/login',  views.login, name = "login"),
    path('api/user/detail', views.get_detail)
]