from django.shortcuts import render
from django.http import HttpResponseRedirect
from .forms import UploadFileForm


def base(request):
    return render(request, 'video/video.html', {})


def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return HttpResponseRedirect('/success/url/')
    else:
        form = UploadFileForm()
    return render(request, 'video/video.html', {'form': form})
