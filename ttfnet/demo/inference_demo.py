from mmdet.apis import init_detector, inference_detector, show_result
import mmcv

config_file = '../configs/ttfnet/ttfnet_d53_2x.py'
checkpoint_file = '../pretrain/ttfnet53_2x-b381dd.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
show_result(img, result, model.CLASSES)

# test a list of images and write the results to image files
imgs = ['test1.jpg', 'test2.jpg']
for i, result in enumerate(inference_detector(model, imgs)):
    show_result(imgs[i], result, model.CLASSES, out_file='result_{}.jpg'.format(i))

# test a video and show the results
video = mmcv.VideoReader('video.mp4')
for frame in video:
    result = inference_detector(model, frame)
    show_result(frame, result, model.CLASSES, wait_time=1)