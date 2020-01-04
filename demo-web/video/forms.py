from django import forms
from .models import Video


class UploadFileForm(forms.ModelForm):
    class Meta:
        model = Video
        fields = ('author', 'title', 'created', 'file')

    def __init__(self, *args, **kwargs):
        self.fields['file'].required = False
