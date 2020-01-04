from django.db import models
from django.utils import timezone


class Video(models.Model):
    author = models.CharField(max_length=5, default="guest")
    title = models.CharField(max_length=500, default="TestVideo")
    created = models.DateTimeField(
        default=timezone.now)
    file = models.FileField(upload_to='', null=True)

    def __str__(self):
        return self.title
