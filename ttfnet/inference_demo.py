from mmdet.apis import init_detector, inference_detector, show_result
import mmcv

config_file = '/home/tjgh131/boaz-adv-project/ttfnet/configs/ttfnet/ttfnet_d53_2x.py'
checkpoint_file = '/home/tjgh131/boaz-adv-project/detection/ttfnet/model/epoch_24.pth'

def main(config_file, checkpoint_file):
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    video = mmcv.VideoReader('test_5.mp4')
    # https://github.com/open-mmlab/mmcv/blob/master/mmcv/video/io.py
    i = 0
    num_person = 0
    num_object = 0
    pre_frame_object = 0
    flag = 0

    for frame in video:
        result = inference_detector(model, frame) #bbox result

        (person_bboxes, object_bboxes) = show_result(frame, result, model.CLASSES, score_thr=0.3,  wait_time=2)
      ##  person_bboxes = list(set(person_bboxes))
      ##  object_bboxes = list(set(object_bboxes))
        # print('person:', person_bboxes)
        # print('len_person :', len(person_bboxes))
        # print('object:', object_bboxes)
        # print('len_object :', len(object_bboxes))
      ##  n_person = len(person_bboxes)
      ##  n_object = len(object_bboxes)
        
        # if person_bboxes != []:
        #     for i in range(n_person):
        #         print('size people height : ',person_bboxes[i][1][1]-person_bboxes[i][0][1])
        #         print('size people width :', person_bboxes[i][1][0] - person_bboxes[i][0][0])
        
       ## if pre_frame_object > n_object: #지금 물체의 개수가 이전 frame보다 줄었다면
       ##     num_object = pre_frame_object #일단 num_object가 원래 물체의 개수로 넣어놓고
       ##     flag = 1 #이때부터 i를 세야한다는 걸 표시
            
       # if n_object == num_object: #다시 돌아오면 초기화
       #     flag = 0
       #     i = 0 
       #     num_object = 0

       # if i >= 15:
       #     print("쓰레기를 버리지 마시오!")
       #     return
                    
       # if flag==1:
       #     i=i+1
        
       # pre_frame_object = n_object
        
if __name__ == '__main__':
    main(config_file, checkpoint_file)
