from django.shortcuts import render
from django.http import HttpResponseRedirect
from .forms import UploadFileForm
import os
from pathlib import Path

from video.violence_model.predict import ViolencePredictor


def base(request):
    return render(request, 'video/base.html', {})


def predict(request):
    BASE = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = Path(BASE, 'violence_model/trained/violence_resnet.model')
    LABEL_PATH = Path(BASE, 'violence_model/trained/lb.pickle')
    INPUT_PATH = os.path.normpath(Path(BASE, '../media/Riot.mp4'))
    OUTPUT_PATH = os.path.normpath(Path(BASE, '../media/output/test1.mp4'))

    ViolencePredictor(model_path=MODEL_PATH, label_path=LABEL_PATH,
                      input_path=INPUT_PATH, output_path=OUTPUT_PATH, size=64)
    return render(request, 'video/base.html', {})


def getExampleVideo(request, exNum):
    return render(request, 'video/base.html', {'path': 'example{}'.format(exNum)})
