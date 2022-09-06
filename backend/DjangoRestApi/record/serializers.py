from rest_framework import serializers 
from record.models import Recording
from users.models import User
from exams.models import Exam
 
class UserSerializer(serializers.ModelSerializer):
    
    class Meta:
        model = User
        fields = ('id', 'username')

class ExamSerializer(serializers.ModelSerializer):
    exams = serializers.PrimaryKeyRelatedField(many=True, read_only=True)
    class Meta:
        model = Exam
        fields = ('id', 'exams')

class RecordingSerializer(serializers.ModelSerializer):
 
    class Meta:
        model = Recording
        fields = ('id',
                'user_id',
                'exam_id',
                'filename',
                'folder_name',
                'created_at')

class RecordingForeignSerializer(serializers.ModelSerializer):   
    user_id = serializers.PrimaryKeyRelatedField(queryset=User.objects.all(),
                                                  many=False)  
    exam_id = serializers.PrimaryKeyRelatedField(queryset=Exam.objects.all(),
                                                  many=False)  
    exam_name = serializers.RelatedField(source='recordings_2.exam_name', read_only=True)
    #exam_name = serializers.RelatedField(queryset=Exam.objects.all(), many = True)
    #exams = ExamSerializer(many=True)
    #exams = serializers.ForeignKey(Exam, related_name='exam_list')
    #exams= ExamSerializer(read_only=True)
    #exam_id = ExamSerializer(many=True)
    #creator_user_id = UserSerializer(many=True)

    class Meta:
        model = Recording
        fields = ('id',
                'user_id',
                'exam_id',
                'filename',
                'folder_name',
                'created_at',
                'exam_name')

