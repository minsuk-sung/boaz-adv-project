from django.urls import path
from . import views

urlpatterns = [
    path('', views.base, name='base'),
    path('example/<int:exNum>', views.getExampleVideo, name='example'),
    path('video_/<int:videoNo>', views.getOutputVideo, name='video_')
]
