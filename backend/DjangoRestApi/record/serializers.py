from rest_framework import serializers 
from record.models import Recording
from users.models import User
 
class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'username')

class RecordingSerializer(serializers.ModelSerializer):
 
    class Meta:
        model = Recording
        fields = ('id',
                'user_id',
                'filename',
                'created_at',
                'timestamp')

class RecordingForeignSerializer(serializers.ModelSerializer):   
    user_id = serializers.PrimaryKeyRelatedField(queryset=User.objects.all(),
                                                  many=False)  
    #exam_id = ExamSerializer(many=True)
    #creator_user_id = UserSerializer(many=True)

    class Meta:
        model = Recording
        fields = ('id',
                'user_id',
                'filename',
                'created_at',
                'timestamp')

