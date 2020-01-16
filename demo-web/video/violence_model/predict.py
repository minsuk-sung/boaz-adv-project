# import os.path
# from keras.models import load_model
# from collections import deque
# import matplotlib.pyplot as plt
# import numpy as np
# import argparse
# import pickle
# import cv2


# def ViolencePredictor(model_path, label_path, input_path, output_path, size):

#     # load the trained model and label binarizer from disk
#     print("[INFO] loading model and label binarizer...")
#     model = load_model(model_path)
#     lb = pickle.loads(open(label_path, "rb").read())
#     print("=============== SUCCESS : LOADING MODEL AND LABEL FILE ===============")

#     # initialize the image mean for mean subtraction along with the
#     # predictions queue
#     mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
#     Q = deque(maxlen=size)

#     # initialize the video stream, pointer to output video file, and
#     # frame dimensions
#     vpath = input_path
#     print("=========", vpath)
#     print("=========", vpath)
#     if input_path == 'camera':
#         vpath = 0
#     vs = cv2.VideoCapture(vpath)
#     writer = None
#     (W, H) = (None, None)

#     # loop over frames from the video file stream
#     while True:
#         # read the next frame from the file
#         (grabbed, frame) = vs.read()

#         # if the frame was not grabbed, then we have reached the end
#         # of the stream
#         if not grabbed:
#             break

#         # if the frame dimensions are empty, grab them
#         if W is None or H is None:
#             (H, W) = frame.shape[:2]

#         # clone the output frame, then convert it from BGR to RGB
#         # ordering, resize the frame to a fixed 224x224, and then
#         # perform mean subtraction
#         output = frame.copy()
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame = cv2.resize(frame, (224, 224)).astype("float32")
#         frame -= mean

#         # make predictions on the frame and then update the predictions
#         # queue
#         preds = model.predict(np.expand_dims(frame, axis=0))[0]
#         Q.append(preds)

#         # perform prediction averaging over the current history of
#         # previous predictions
#         results = np.array(Q).mean(axis=0)
#         # i = np.argmax(results)
#         i = 1
#         label = lb.classes_[i]

#         # draw the activity on the output frame
#         # prob = model.predict_proba(np.expand_dims(frame, axis=0))[0] # to show probability of frame
#         prob = results[i]*100

#         text_color = (0, 255, 0)  # default : green

#         if prob > 70:  # Violence prob
#             text_color = (0, 0, 255)  # red

#         else:
#             label = 'Normal'

#         text = "State : {:8} ({:3.2f}%)".format(label, prob)
#         FONT = cv2.FONT_HERSHEY_SIMPLEX

#         cv2.putText(output, text, (35, 50), FONT, 1.25, text_color, 3)

#         # plot graph over background image
#         output = cv2.rectangle(
#             output, (35, 80), (35+int(prob)*5, 80+20), text_color, -1)

#         # check if the video writer is None
#         if writer is None:
#             # initialize our video writer
#             fourcc = cv2.VideoWriter_fourcc(*"MJPG")
#             writer = cv2.VideoWriter(output_path, fourcc, 30, (W, H), True)

#         # write the output frame to disk
#         writer.write(output)

#         # show the output image
#         cv2.imshow("Output", output)
#         key = cv2.waitKey(1) & 0xFF

#         # if the `q` key was pressed, break from the loop
#         if key == ord("q"):
#             break

#     # release the file pointers
#     print("[INFO] cleaning up...")
#     writer.release()
#     vs.release()
