from django.shortcuts import render
from django.http import HttpResponseRedirect
from .forms import UploadFileForm
from .apps import VideoConfig


def base(request):
    return render(request, 'video/base.html', {})


def getExampleVideo(request, exNum):
    return render(request, 'video/base.html', {'path': 'example{}'.format(exNum)})
