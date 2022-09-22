from rest_framework import serializers 
from exams.models import Exam
from users.models import User



class ExamSerializer(serializers.ModelSerializer):
    class Meta:
        model = Exam
        fields = ('id',
                'exam_name',
                'analyze',
                'status',
                'created_at',
                'last_modified_at')



class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id',
                'username',
                'password',
                'is_superuser',
                'is_active',
                'contract_signed',
                'created_at',
                'last_modified_at')
 


class ExamForeignSerializer(serializers.ModelSerializer):    
    class Meta:
        model = Exam
        fields = ('id',
                'exam_name',
                'status',
                'creator_user_id',
                'created_at',
                'last_modified_at')
