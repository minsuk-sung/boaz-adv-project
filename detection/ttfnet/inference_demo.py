from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="video inference")
    parser.add_argument('--config', required=False, default='/home/kdh/ttfnet2/configs/ttfnet/ttfnet_d53_1x.py', help="config file path")
    parser.add_argument("--checkpoint", required=False, default="/home/kdh/work_dir/seo-ho2.pth")
    parser.add_argument("--video", required=False, default="/home/kdh/V_99.mp4")
    args = parser.parse_args()
    return args
#config_file = '/home/kdh/ttfnet2/configs/ttfnet/ttfnet_d53_1x.py'
#checkpoint_file = '/home/kdh/ttfnet2/work_dir/ttfnet53_1x/latest.pth'
#checkpoint_file = "/home/kdh/work_dir/seo-ho2.pth"

def main():
    args = parse_args()
    config_file = args.config
    checkpoint_file = args.checkpoint
    video_path = args.video
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # test a single image and show the results
    # img = 'image11945.jpg'  # or img = mmcv.imread(img), which will only load it once
    # result = inference_detector(model, img)
    # show_result(img, result, model.CLASSES)

    # # test a list of images and write the results to image files
    # imgs = ['test1.jpg', 'test2.jpg']
    # for i, result in enumerate(inference_detector(model, imgs)):
    #     show_result(imgs[i], result, model.CLASSES, out_file='result_{}.jpg'.format(i))

    # test a video and show the results
    video = mmcv.VideoReader(video_path) #('video.mp4')
    # https://github.com/open-mmlab/mmcv/blob/master/mmcv/video/io.py
    print('len(video):',len(video)) #129 - 5초 / 74 - 2초

    for frame in video:
        result = inference_detector(model, frame)
        show_result(frame, result, model.CLASSES, wait_time=1)
        
if __name__ == '__main__':
    main()
