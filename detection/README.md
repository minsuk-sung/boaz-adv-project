# VIOLENCE ACTION DETECTION

학습시킨 모델은 구글 드라이브 detection_model$epoch_24.pth로 올려둠

----

### DATA

Classfication과 마찬가지로 Kaggle 데이터 사용

> https://www.kaggle.com/mohamedmustafa/real-life-violence-situations-dataset

----

### MODEL

2019년 8월에 나온 TTFNet 사용(당시 기준으로 SOTA)

> https://www.kaggle.com/mohamedmustafa/real-life-violence-situations-dataset

----

### Environment

TTFNet의 INSTALL.md를 참고했음.

GPU: K80
CUDA: 10.0.130
CUDNN: 7.6.5
Python: 3.7
pytorch = 1.2.0
torchvision = 0.4.0

----

### TTFNet Installation

```bash
# 콘다 가상환경 만들기
conda create -n ttfnet python=3.7 -y
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
conda install cython
pip install mmcv
python3 setup.py develop
* python3로 사용하는 이유는 cc1plus warning 방지하기 위함

# import 코드 수정
ttfnet/mmdet/datasets/loader/sampler.py
mmcv.runner.utils -> mmcv.runner 바꾸기
* 안하면 import error 발생

# config 파일 수정
vi ttfnet/configs/ttfnet/ttfnet_d53_1x.py
1) pretrained 경로 수정
- pretrained darknet 다운 받아야 함
- wget http://downloads.zjulearning.org.cn/ttfnet/darknet53_pretrain-9ec35d.pth
> https://github.com/ZJULearning/ttfnet 사이트의 Inference 밑에 부분 보면 다운 링크 있음.
2) data_root 경로 수정
- custom data가 있는 경로
3) train, val, test의 ann_file, img_prefix 경로 수정
- ann_file은 coco 형식의 json 파일
- img_prefix는 image가 들어있는 directory
4) load_from 수정
- pretrained ttfnet 다운 받아야 함
- wget http://downloads.zjulearning.org.cn/ttfnet/ttf53_aug_10x-86c43dd3.pth
5) work_dir 수정
- train 하고 나서 생기는 log파일, pth파일 저장할 디렉토리

# inference 파일 수정
vi ttfnet/inference.py
- parser에서 config의 default를 위에서 수정한 config 경로로 수정
- parser에서 checkpoint의 default를 train 후 생성된 pth 파일로 경로 수정
- parser에서 video의 default를 테스트 할 video path로 수정
* 경로 수정 하지 않고 parsing만 해도 동작 하긴 함.

# custom class 사용을 위한 파일 수정
1) vi ~/ttfnet/mmdet/apis/inference.py
if checkpoint is not None:
	checkpoint = load_checkpoint(model, checkpoint)
	if 'CLASSES' in checkpoint['meta']:
		model.CLASSES = get_classes('own')
		# model.CLASSES = checkpoint['meta']['CLASSES']
2) vi ~/ttfnet/mmdet/core/evaluation/class_names.py
# 새로 생성
def own_classes():
    return[
            "fight"]

dataset_aliases = {
    'voc': ['voc', 'pascal_voc', 'voc07', 'voc12'],
    'imagenet_det': ['det', 'imagenet_det', 'ilsvrc_det'],
    'imagenet_vid': ['vid', 'imagenet_vid', 'ilsvrc_vid'],
    'own' : ['own'], # 추가한 부분
    'coco': ['coco', 'mscoco', 'ms_coco'],
    'wider_face': ['WIDERFaceDataset', 'wider_face', 'WDIERFace'],
    'cityscapes': ['cityscapes']
}
3) vi ~/ttfnet/mmdet/datasets/my_dataset.py
from .coco import CocoDataset
from .registry import DATASETS

@DATASETS.register_module
class MyDataset(CocoDataset):

    CLASSES = ("fight")
4) vi ~/ttfnet/mmdet/datasets/custom.py
CLASSES = None -> 'fight' 변경

5) vi ~/ttfnet/mmdet/datasets/__init__.py
from .my_dataset import MyDataset 추가
__all__ 리스트에 "MyDataset" 추가

# train
python tools/train.py [config 파일 경로]

# test
python inference_demo.py --config=[config 파일 경로] --checkpoint=[checkpoint 경로] --video=[video 경로]
```

----

### GCP 사용자를 위해

#### GCP 접속 설정

```bash
# local에서 ssh key pair 만들기
cd ~/.ssh
ssh-keygen -t rsa -C [계정이름]
// -t 옵션은 어떤 암호화 알고리즘을 사용할 것인지에 대한 것. 기본값은 rsa
// -b 옵션은 몇 bit key를 만들 것인지 결정하는 옵션. 기본값은 2048
~/.ssh에서 id_rsa.pub 내용 복사

# 참고
pub = 열쇠
rsa = 자물쇠

# gcp 계정에 ssh key pair 복사하기
메타테이터에 붙여넣기

# ssh로 접속(계정 이름, 비밀번호는 key pair만들 때 사용했던 것
ssh -i [key pair = rsa 파일] [계정 이름]@[주소]
<예시>
ssh -i ~/.ssh/kdh kdh@35.185.223.52

<오류> vi: command not found
# vim 설치
sudo apt-get update
sudo apt-get install vim

# 사용자 목록
vi /etc/passwd

# ssh 비밀번호 접속 허용(이번에는 이미 yes로 바뀌어있음)
sudo vi /etc/ssh/sshd_config
PasswordAuthentication no -> yes

# scp로 파일 보내기(publickey 필요하면 -i 옵션 파일 뒤에 사용)
scp 파일 계정@서버주소:목적경로
scp -r 디렉토리 계정@서버주소:목적경로
<예시> - publickey 사용해서 디렉토리 보내기
scp -i ~/.ssh/kdh -r /Users/KDH/Downloads/cudnn kdh@35.185.223.52:/home/kdh


# 참고
## 사용자 추가
sudo useradd [사용자 이름]  # 계정 ID만 생성(홈디렉토리 등 설정 X)
sudo adduser [사용자 이름]  # 계정 ID 및 홈디렉토리, 계정정보 및 비밀번호 등 기본으로 설정
sudo passwd [사용자 이름]

# reference
http://dev.crois.net/2018/02/27/cloud-google-cloud-ssh-key-이용하여-접속/
https://noota.tistory.com/entry/SCP-리눅스-터미널-환경에서-다른서버의-파일-복사해오기
https://zetawiki.com/wiki/리눅스_scp_사용법
```

#### TTFNET을 위한 기본 설치

```bash
# 1. gcc 설치
sudo apt install gcc


# 2. make 설치
sudo apt install make


# 3. nvidia driver 설치
sudo apt-get update
sudo apt install ubuntu-drivers-common
ubuntu-drivers devices
sudo apt-get install nvidia-driver-435
sudo reboot

## nvidia driver 설치 확인
nvidia-smi

## Reference
https://doctorson0309.tistory.com/513
https://codechacha.com/ko/install-nvidia-driver-ubuntu/


# 4. cuda 설치
sudo apt install nvidia-cuda-toolkit
wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux
wget http://developer.download.nvidia.com/compute/cuda/10.0/Prod/patches/1/cuda_10.0.130.1_linux.run

## base 설치
sudo sh cuda_10.0.130_410.48_linux

## patch설치
sudo sh cuda_10.0.130.1_linux.run

## 설치과정
==============================
긴 글은 ctrl + c로 스킵 가능
Do you accept the previously read EULA? accept/decline/quit: **accept**
Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 430.40? (y)es/(n)o/(q)uit: **n**
Install the CUDA 10.0 Toolkit? (y)es/(n)o/(q)uit: **y**
Enter Toolkit Location [ default is /usr/local/cuda-8.0 ]: **엔터 입력**
Do you want to install a symbolic link at /usr/local/cuda? (y)es/(n)o/(q)uit: **y**
Install the CUDA 10.0 Samples? (y)es/(n)o/(q)uit: **y**
Enter CUDA Samples Location [ default is /home/python-kim ]: **엔터 입력**
====================================
## 경로 추가
export CUDA_HOME=/usr/local/cuda-10.0
export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

## cuda 설치 확인
nvcc --version

## Reference
https://blog.naver.com/PostView.nhn?blogId=angelkim88&logNo=221630554860&categoryNo=138&parentCategoryNo=0&viewDate=&currentPage=1&postListTopCurrentPage=1&from=search


# 5. cudnn 설치
[로컬에서 파일 3개 다운 받아서 서버로 파일 옮겨놓기]
sudo dpkg -i libcudnn7_7.6.5.32-1+cuda10.0_amd64.deb
sudo dpkg -i libcudnn7-dev_7.6.5.32-1+cuda10.0_amd64.deb
sudo dpkg -i libcudnn7-doc_7.6.5.32-1+cuda10.0_amd64.deb

## cudnn 설치 확인
cat /usr/include/x86_64-linux-gnu/cudnn_v*.h | grep CUDNN_MAJOR -A 2

## Reference
[먼저 참고] https://ruuci.tistory.com/tag/cudnn
https://blog.naver.com/PostView.nhn?blogId=angelkim88&logNo=221630569516&parentCategoryNo=&categoryNo=138&viewDate=&isShowPopularPosts=true&from=search


# 6. 아나콘다 설치
wget https://repo.continuum.io/archive/Anaconda3-5.3.1-Linux-x86_64.sh
[참고] 버전 바꾸고 싶으면 아카이브에서 찾아보기
bash Anaconda3-5.3.1-Linux-x86_64.sh
source ~/.bashrc

## 오류
### conda: command not found
bash 설정을 안해줘서 그럼
[일회용] source /home/KDH/anaconda3/etc/profile.d/conda.sh
[영구적] vi ~/.bashrc에 export PATH="~/anaconda3/bin:$PATH" 추가하고, source로 새로고침

## Reference
https://shwksl101.github.io/gcp/2018/12/23/gcp_vm_custom_setting.html


# 7. remote jupyter notebook local에서 사용
## server
[생략 해도 되는 듯] conda install -c anaconda jupyter
pip install ipykernel
python -m ipykernel install --user --name [virtualEnv] --display-name [displayKenrelName]

jupyter notebook --generate-config
vi ~/.jupyter/jupyter_notebook_config.py
[내용 추가]
c = get_config()
c.NotebookApp.ip = "[external IP]"
c.NotebookApp.open_browser = False
c.NotebookApp.port = 8888 # (gcp 인스턴스에서 방화벽 만들었을 때 설정했던 포트)

jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root

## local
[external IP]:8888
초기 token은 server terminal에 적혀있음(token=뒤에 부분 복사)
비밀번호 설정 후 이용

## 자동 실행 참고
https://zzsza.github.io/gcp/2018/06/14/install-tensorflow-jupyter-in-gcp/
```