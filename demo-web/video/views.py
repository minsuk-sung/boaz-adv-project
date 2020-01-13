from django.shortcuts import render
from django.http import HttpResponseRedirect
from .forms import UploadFileForm


def base(request):
    return render(request, 'video/video.html', {})
