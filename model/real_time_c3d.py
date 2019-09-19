#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2018/6/8 21:11
     # @Author  : Awiny
     # @Site    :
     # @Project : C3D-tensorflow
     # @File    : real_time_c3d.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning

import time
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import c3d_model
from real_time_input_data import *
import numpy as np
import cv2
import heapq

# Basic model parameters as external flags.
flags = tf.app.flags
gpu_num = 1
flags.DEFINE_integer('batch_size', 1 , 'Batch size.')
FLAGS = flags.FLAGS

images_placeholder = tf.placeholder(tf.float32, shape=(1, 16, 112, 112, 3))
labels_placeholder = tf.placeholder(tf.int64, shape=1)

def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.
  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.
  Args:
    batch_size: The batch size will be baked into both placeholders.
  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(1, 16,112,112,3))
  labels_placeholder = tf.placeholder(tf.int64, shape=1)
  return images_placeholder, labels_placeholder

# !! 질문 !! 어떤 함수인지 모르겠음.
# cpu 디바이스 정해주는 건가 ..
def _variable_on_cpu(name, shape, initializer):
    #with tf.device('/cpu:%d' % cpu_id):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

# weight 계산해주는 함수
def _variable_with_weight_decay(name, shape, stddev, wd):
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    # !! 도와주세요 !! 알 수 없어요
    if wd is not None:
        weight_decay = tf.nn.l2_loss(var) * wd # L2 loss 통해 weight decay 계산
        tf.add_to_collection('losses', weight_decay)
    return var

# 예측값 도출해주는 함수 : top1, top5
def run_one_sample(norm_score, sess, video_imgs):
    """
    run_one_sample and get the classification result
    :param norm_score:
    :param sess:
    :param video_imgs:
    :return:
    """
# images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size * gpu_num)
#    start_time = time.time()
#    video_imgs = np.random.rand(1, 16, 112, 112, 3).astype(np.float32)
    predict_score = norm_score.eval(
            session=sess,
            feed_dict={images_placeholder: video_imgs} # 16개 프레임 이미지들 받아온 것
            )
    top1_predicted_label = np.argmax(predict_score) # top1
    predict_score = np.reshape(predict_score,101)
    #print(predict_score)
    top5_predicted_value = heapq.nlargest(5, predict_score) # top5 value
    top5_predicted_label = predict_score.argsort()[-5:][::-1] # top5 label
    return top1_predicted_label, top5_predicted_label, top5_predicted_value # top1이랑, top5들 리턴

# pre-train된 model을 돌려서 그 결과 (norm_score, sess)를 리턴하는 함수이다.
def build_c3d_model():
    """
    build c3d model
    :return:
    norm_score:
    sess:
    """
    #model_name = "pretrained_model/c3d_ucf101_finetune_whole_iter_20000_TF.model.mdlp"
    #model_name = "pretrained_model/conv3d_deepnetA_sport1m_iter_1900000_TF.model"
    model_name = "pretrained_model/sports1m_finetuning_ucf101.model"
    # Get the sets of images and labels for training, validation, and
    # 논문에 나와있는대로 값 넣어서 weights, biases 구함
    with tf.variable_scope('var_name') as var_scope:
        weights = {
            'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.04, 0.00),
            'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.04, 0.00),
            'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.04, 0.00),
            'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.04, 0.00),
            'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.04, 0.00),
            'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.04, 0.001),
            'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.04, 0.002),
            'out': _variable_with_weight_decay('wout', [4096, c3d_model.NUM_CLASSES], 0.04, 0.005)
        }
        biases = {
            'bc1': _variable_with_weight_decay('bc1', [64], 0.04, 0.0),
            'bc2': _variable_with_weight_decay('bc2', [128], 0.04, 0.0),
            'bc3a': _variable_with_weight_decay('bc3a', [256], 0.04, 0.0),
            'bc3b': _variable_with_weight_decay('bc3b', [256], 0.04, 0.0),
            'bc4a': _variable_with_weight_decay('bc4a', [512], 0.04, 0.0),
            'bc4b': _variable_with_weight_decay('bc4b', [512], 0.04, 0.0),
            'bc5a': _variable_with_weight_decay('bc5a', [512], 0.04, 0.0),
            'bc5b': _variable_with_weight_decay('bc5b', [512], 0.04, 0.0),
            'bd1': _variable_with_weight_decay('bd1', [4096], 0.04, 0.0),
            'bd2': _variable_with_weight_decay('bd2', [4096], 0.04, 0.0),
            'out': _variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], 0.04, 0.0),
        }
    logits = []
    for gpu_index in range(0, gpu_num):
        with tf.device('/gpu:%d' % gpu_index):
            # c3d_model.py 의 inference_c3d 함수에 weight랑 bias 넣어서 돌리고. 그 결과가 logits
            logit = c3d_model.inference_c3d(
                images_placeholder[0 * FLAGS.batch_size:(0 + 1) * FLAGS.batch_size,:,:,:,:], 0.6,
                FLAGS.batch_size, weights, biases)
            logits.append(logit)
    logits = tf.concat(logits, 0) # axis=0으로 concat
    norm_score = tf.nn.softmax(logits) # logits softmax 돌려서 score 저장
    saver = tf.train.Saver() # 모델과 파라미터 saver에 저장
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # 명시된 디바이스가 존재하지 않는 경우 실행 디바이스를 TensorFlow가 자동으로 존재하는 디바이스 중 선택하게 하려면,
    # 세션을 만들 때 allow_soft_placement 옵션을 True로 설정하면 된다고 한다.
    init = tf.global_variables_initializer()
    sess.run(init)
    # Create a saver for writing training checkpoints.
    saver.restore(sess, model_name) # 저장된 모델과 파라미터 불러옴 (Restore)
    return norm_score, sess # score 계산한거랑 session 리턴

# 실시간으로 비디오를 받아 classification 을 진행하는 함수이다.
def real_time_recognition(video_path):
    """
    real time video classification
    :param video_path:the origin video_path
    :return:
    """
    norm_score, sess = build_c3d_model() # pretrained를 c3d 모델 돌려서 나온 score랑 session
    video_src = video_path if video_path else 0 # 비디오 패쓰 받음 (0:웹캠)
    cap = cv2.VideoCapture(video_src) # 비디오 패쓰 받아온 것으로 비디오 캡쳐 객체 생성
    count = 0
    video_imgs = []
    predicted_label_top5 = []
    top5_predicted_value = []
    predicted_label = 0
    classes = {}
    flag = False
    with open('./list/classInd.txt', 'r') as f: # classlist 열어따
        for line in f:
            content = line.strip('\r\n').split(' ')
            classes[content[0]] = content[1] # classes라는 배열에 list에서 받아온 class들 저장
            # classes[1] = ApplyEyeMakeup 이런식인듯.
   # print(classes)
    while True:
        ret, img = cap.read() # 비디오 한 프레임씩 읽는다.
        # 제대로 읽으면 ret=True, 실패하면 False.
        # img : 읽은 프레임이 담긴다.
        if type(img) == type(None): # 프레임 안 읽히면 끝내야지
            break
        float_img = img.astype(np.float32) # 이미지를 float로 바꿨네용
        video_imgs.append(float_img) # video_imgs 배열에 넣었네용
        count += 1
        if count == 16: # 16 프레임을 다 받으면
            video_imgs_tensor = clip_images_to_tensor(video_imgs, 16, 112)
            # real_time_input_data.py 에 있는 함수 : 이미지 크롭하고 전처리 해서 리턴 (112x112)에 맞게.
            predicted_label, predicted_label_top5, top5_predicted_value = run_one_sample(norm_score, sess, video_imgs_tensor)
            # 예측값, top5를 도출 : run_one_sample 함수
            count = 0 # 다시 초기화
            video_imgs = [] # 다시 초기화
            flag = True # 액션 분류 해냈다 ~~~
            # 근데 flag 다시 false 해주는 코드는 안 써도 되는건가 ,,
          # channel_1, channel_2, channel_3 = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        if flag:
            for i in range(5):
                cv2.putText(img, str(top5_predicted_value[i])+ ' : ' + classes[str(predicted_label_top5[i] + 1)], (10, 15*(i+1)),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                            1, False) # 예측한 라벨 보일 수 있도록 putText

        cv2.imshow('video', img) # img (읽은 프레임) 보여줌

        if cv2.waitKey(33) == 27: # key 눌러서 (ESC누를 경우 27 리턴) 나가고 싶을 때 나가기
            break

    cv2.destroyAllWindows() # 윈도우 종료

# video_path를 input으로 받아 실시간으로 classification을 진행하는 방식이다.
# video_path를 받으면, real_time_recognition 함수를 실행시켜 classification을 진행한다.
def main(_):
    video_path = input("please input the video path to be classification:")
    real_time_recognition(video_path)

if __name__ == '__main__':
    tf.app.run()
