from django.db import models
import django.db.models.deletion


#exam model
class Exam(models.Model):

    TTL_STATUS = (
        (0, 'Inactive'),
        (1, 'Active'),
    )

    TTL_ANALYZE = (
        (0, 'No'),
        (1, 'Yes'),
    )
    id = models.AutoField(primary_key=True, auto_created=True)
    exam_name = models.CharField(max_length=100)
    status = models.SmallIntegerField(choices=TTL_STATUS, default=0)
    analyze = models.SmallIntegerField(choices=TTL_ANALYZE, default=0)
    created_at = models.DateTimeField(auto_now_add=True, editable=False)
    last_modified_at = models.DateTimeField(auto_now=True, editable=False)

    def __str__(self):
        return self.exam_name
