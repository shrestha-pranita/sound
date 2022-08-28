from django.db import models
import django.db.models.deletion
import os

class Recording(models.Model):
    TTL_STATUS = (
        (0, 'No'),
        (1, 'Yes'),
    )

    id = models.AutoField(primary_key=True, auto_created=True)
    user_id = models.ForeignKey(
        on_delete=django.db.models.deletion.CASCADE, related_name='recordings_3', to='users.User', )
    filename = models.CharField(max_length=200, blank=False, default='')
    created_at = models.DateTimeField(auto_now_add=True, editable=False)
    timestamp = models.SmallIntegerField(choices=TTL_STATUS, default=0)