from django.apps import AppConfig
from pathlib import Path
import os.path

from video.violence_model.predict import ViolencePredictor


class VideoConfig(AppConfig):
    name = 'video'

    BASE = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = Path(BASE, 'violence_model/trained/violence_resnet.model')
    LABEL_PATH = Path(BASE, 'violence_model/trained/lb.pickle')
    INPUT_PATH = os.path.normpath(Path(BASE, '../media/street-fight.mp4'))
    OUTPUT_PATH = os.path.normpath(Path(BASE, '../media/output/'))

    # predictor = ViolencePredictor(model_path=MODEL_PATH, label_path=LABEL_PATH,
    #                               input_path=INPUT_PATH, output_path=OUTPUT_PATH, size=64)
