from django.db import models
from django.contrib.auth.models import AbstractUser
from django.contrib.auth.models import User

class User(AbstractUser):

    TTL_SUPERUSER = (
        (0, 'No'),
        (1, 'Yes'),
    )
    """
    TTL_ACTIVE = (
        (0, 'Inactive'),
        (1, 'Active'),
    )

    TTL_SIGNED = (
        (0, 'No'),
        (1, 'Yes'),
    )
    """
    id = models.AutoField(primary_key=True, auto_created=True)
    username = models.CharField(max_length=100,unique=True)
    is_active = models.BooleanField(default=True)
    
    email = models.EmailField(unique=True, null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True, editable=False)
    last_modified_at = models.DateTimeField(auto_now=True, editable=False)