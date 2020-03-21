from django.db import models
from django.contrib.auth.models import User


class Activation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    code = models.CharField(max_length=20, unique=True)
    email = models.EmailField(blank=True)

from django.core.validators import RegexValidator
class OnlineUsers(models.Model):
    phone_regex = RegexValidator(regex=r'^\+?1?\d{6,10}$',
                                 message="Phone number must be entered in the format: '+999999999'. Up to 10 digits allowed.")

    name=models.CharField(max_length=100)
    contact_no = models.CharField(validators=[phone_regex],max_length=10)
    latitude = models.FloatField(default=0.0)
    longitude = models.FloatField(default=0.0)
    email = models.EmailField(unique=True,max_length=50)
    usertype = models.CharField( default='user',max_length=10)
    password = models.CharField(max_length=30)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name
    class Meta:
        db_table = "online_users"

