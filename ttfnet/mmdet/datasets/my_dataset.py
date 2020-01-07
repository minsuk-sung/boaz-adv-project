# from .xml_style import XMLDataset
# from pycocotools.coco import COCO
from .registry import DATASETS
from .coco import CocoDataset

@DATASETS.register_module
class MyDataset(CocoDataset):

    CLASSES = ('fight')
