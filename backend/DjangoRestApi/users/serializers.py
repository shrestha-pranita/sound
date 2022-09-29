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
                'created_at',
                'last_modified_at')


#login Serializer 

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




#forgetpassword serializer
class PasswordSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField()

    def create(self, validated_data):
        pass

    def update(self, instance, validated_data):
        pass

    def validate(self, data):
        """ check that username and new password are different """
        if data["username"] == data["password"]:
            raise serializers.ValidationError("Username and new password should be different")
        return data

    def validate_password(self, value):
        """
        check if new password meets the specs
        min 1 lowercase and 1 uppercase alphabet
        1 number
        1 special character
        8-16 character length
        """

        # if len(value) < 8 or len(value) > 16:
        #     raise serializers.ValidationError("It should be between 8 and 16 characters long")

        # if not any(x.isupper() for x in value):
        #     raise serializers.ValidationError("It should have at least one upper case alphabet")

        # if not any(x.islower() for x in value):
        #     raise serializers.ValidationError("It should have at least one lower case alphabet")

        # if not any(x.isdigit() for x in value):
        #     raise serializers.ValidationError("It should have at least one number")

        # valid_special_characters = {'@', '_', '!', '#', '$', '%', '^', '&', '*', '(', ')',
        #                             '<', '>', '?', '/', '|', '{', '}', '~', ':'}

        # if not any(x in valid_special_characters for x in value):
        #     raise serializers.ValidationError("It should have at least one special character")

        return value
