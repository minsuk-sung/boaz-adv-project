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
from keras.models import load_model
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, default='model/violence.model',
	help="path to trained serialized model")
ap.add_argument("-l", "--label-bin", required=True, default='model/lb.pickle',
	help="path to  label binarizer")
ap.add_argument("-i", "--input", required=True, default='example_clips/lifting.mp4',
	help="path to our input video")
ap.add_argument("-o", "--output", required=True, default='output/lifting_128avg.avi',
	help="path to our output video")
ap.add_argument("-s", "--size", type=int, default=128,
	help="size of queue for averaging")
args = vars(ap.parse_args())

# load the trained model and label binarizer from disk
print("[INFO] loading model and label binarizer...")
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())

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
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

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
	
	else:
		label = 'Normal'

	text = "State : {:8} ({:3.2f}%)".format(label,prob)
	FONT = cv2.FONT_HERSHEY_SIMPLEX 

	cv2.putText(output, text, (35, 50), FONT,1.25, text_color, 3) 

	# plot graph over background image
	output = cv2.rectangle(output, (35, 80), (35+int(prob)*5,80+20), text_color,-1)

	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,(W, H), True)

	# write the output frame to disk
	writer.write(output)

	# show the output image
	cv2.imshow("Output", output)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
