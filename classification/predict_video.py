# Source
# https://www.pyimagesearch.com/2019/07/15/video-classification-with-keras-and-deep-learning/
# https://www.pyimagesearch.com/2019/05/20/transfer-learning-with-keras-and-deep-learning/

# Grad CAM
# https://github.com/you359/Keras-CNNVisualization/tree/master/keras-GradCAM
# https://github.com/jacobgil/keras-grad-cam

# USAGE
# python predict_video.py --model model/violence_mobile.model --label-bin model/lb.pickle --input example_clips/lifting.mp4 --output output/lifting_128avg.avi --size 64
# python predict_video.py --model model/violence_mobile.model --label-bin model/lb.pickle --input camera --output real_time.avi --size 64


# import the necessary packages
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import os

def get_session(gpu_fraction=0.7):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())
from keras.models import load_model
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import time


# ttfnet
from mmdet.apis import init_detector, inference_detector, show_result
import mmcv




# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=False, default='./model/violence_resnet.model',
	help="path to trained serialized model")
ap.add_argument("-l", "--label-bin", required=False, default='./model/lb.pickle',
	help="path to  label binarizer")
ap.add_argument("-i", "--input", required=False, default='./video/news.mp4',
	help="path to our input video")
ap.add_argument("-o", "--output", required=False, default='./output/test.avi',
	help="path to our output video")
ap.add_argument("-s", "--size", type=int, default=60,
	help="size of queue for averaging")
args = vars(ap.parse_args())

# load the trained model and label binarizer from disk
print("[INFO] loading model and label binarizer...")
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())


# ttfnet
config_file='/home/tjgh131/boaz-adv-project/ttfnet/configs/ttfnet/ttfnet_d53_2x.py'
checkpoint_file = '/home/tjgh131/boaz-adv-project/detection/ttfnet/model/epoch_24.pth'

model_detect = init_detector(config_file, checkpoint_file, device='cuda:0')

# initialize the image mean for mean subtraction along with the
# predictions queue
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=args["size"])

# initialize the video stream, pointer to output video file, and
# frame dimensions
vpath = args["input"]
if args["input"] == 'camera':
	vpath = 0
vs = cv2.VideoCapture(vpath)
writer = None
(W, H) = (None, None)

# loop over frames from the video file stream
while True:
	# read the next frame from the file
    fps_time = time.time()
    (grabbed, frame) = vs.read()
    
    if not grabbed:
        break

        #ttfnet
    result_detect = inference_detector(model_detect, frame)

	# if the frame was not grabbed, then we have reached the end
	# of the stream


	# if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

	# clone the output frame, then convert it from BGR to RGB
	# ordering, resize the frame to a fixed 224x224, and then
	# perform mean subtraction
    
    output = frame.copy()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224)).astype("float32")
    frame -= mean
    
	# make predictions on the frame and then update the predictions
	# queue
    preds = model.predict(np.expand_dims(frame, axis=0))[0]
    Q.append(preds)

        

	# perform prediction averaging over the current history of
	# previous predictions
    results = np.array(Q).mean(axis=0)
	# i = np.argmax(results)
    i = 1
    label = lb.classes_[i]
	

	# draw the activity on the output frame
	# prob = model.predict_proba(np.expand_dims(frame, axis=0))[0] # to show probability of frame
    prob = results[i]*100

    text_color = (0, 255, 0) # default : green

    if prob > 70 : # Violence prob
        text_color = (0, 0, 255) # red
                # ttfnet
        (person_bboxes, object_bboxes, image) = show_result(output, result_detect, model_detect.CLASSES, score_thr=0.3, wait_time=2)
        output = image
	
    else:
        label = 'Normal'

    text = "State : {:8} ({:3.2f}%)".format(label,prob)
    FONT = cv2.FONT_HERSHEY_SIMPLEX 

   # cv2.putText(output, text, (35, 50), FONT,1.25, text_color, 3) 
    cv2.putText(output, text, (35, 30), FONT, 0.75, text_color, 3)

	# plot graph over background image
   # output = cv2.rectangle(output, (35, 80), (35+int(prob)*5,80+20), text_color,-1)
    output = cv2.rectangle(output, (35, 50), (300, 60), text_color, -1)

	# check if the video writer is None
    if writer is None:
		# initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,(W, H), True)

	# show the output image
    cv2.putText(output, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    writer.write(output)
    cv2.imshow("Output", output)
    key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
