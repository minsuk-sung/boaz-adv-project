from django.db import models
from django.utils import timezone


class Video(models.Model):
    author = models.CharField(max_length=5, default="guest")
    title = models.TextField(default="TestVideo")
    created = models.DateTimeField(
        default=timezone.now)
    # is_violence = models.BooleanField(null=True)

    def __str__(self):
        return self.title
