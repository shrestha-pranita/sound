from django.db import models
import django.db.models.deletion
import os
from exams.models import Exam

class Recording(models.Model):
    TTL_STATUS = (
        (0, 'No'),
        (1, 'Yes'),
    )

    TTL_ANALYZE = (
        (0, 'No'),
        (1, 'Yes'),
    )

    id = models.AutoField(primary_key=True, auto_created=True)
    user_id = models.ForeignKey(
        on_delete=django.db.models.deletion.CASCADE, related_name='recordings_1', to='users.User', )
    exam_id = models.ForeignKey(
        on_delete=django.db.models.deletion.CASCADE, related_name='recordings_2', to='exams.Exam', )
    speech_detected = models.IntegerField(default=0)
    speech_analyzed = models.SmallIntegerField(choices=TTL_ANALYZE, default=0)
    filename = models.CharField(max_length=200, blank=False, default='')
    folder_name = models.CharField(max_length=200, blank=False, default='')
    created_at = models.DateTimeField(auto_now_add=True, editable=False)


    @property
    def exam_name(self):
        return self.exam_id.exam_name