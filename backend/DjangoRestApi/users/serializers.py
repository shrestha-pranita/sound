from rest_framework import serializers 
from users.models import User
from django.contrib.auth.hashers import make_password
 
 
class UserSerializer(serializers.ModelSerializer):
 
    class Meta:
        model = User
        fields = ('id',
                'username',
                'password',
                'is_superuser',
                'is_active',
                #'contract_signed',
                'created_at',
                'last_modified_at')

class LoginSerializer(serializers.Serializer):  
    username = serializers.CharField(label="Username", required=True)
    password = serializers.CharField(label="Password", required=True)

# Register serializer
class RegistrationSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id','email','password', 'username')
        extra_kwargs = {
            'password':{'write_only': True},
        }
    def create(self, validated_data):
        user = User.objects.create_user(email=validated_data['email'],  
                                        username=validated_data['username'] ,
        password = validated_data['password'])
        return user
