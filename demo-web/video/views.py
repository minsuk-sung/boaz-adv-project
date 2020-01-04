from django.shortcuts import render


def base(request):
    return render(request, 'video/upload_video.html', {})
