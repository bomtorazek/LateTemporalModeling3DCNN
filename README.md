# LateTemporalModeling3DCNN

Pytorch implementation of [Late Temporal Modeling in 3D CNN Architectures with BERT for Action Recognition]

This the repository which implements late temporal modeling on top of the 3D CNN architectures and mainly focus on BERT for this aim. 

## Installation
	#For the installation, you need to install conda. The environment may contain also unnecessary packages but we want to give complete environment that we are using. 

	#Environment by esuh
	conda env create -f new_LateTemporalModeling3D.yml

	#Then you can activate the environment with the command
	conda activate 3dcnn

Later, please download the necessary files from the link, and copy them into the main directory.

https://1drv.ms/u/s!AqKP51Rjkz1Gaifd54VbdRBn6qM?e=7OxYLa

위의 링크에서 다운로드가 안 되시면 [해당 경로](\\10.99.160.32\Archive-Research\_temp\eungyosuh)에 제가 올려놨으니 다운 부탁드립니다.

왠지는 모르겟지만 r2plus1d에 대한 weight가 없어서 [IG-65M unofficial repo](https://github.com/moabitcoin/ig65m-pytorch)에서

[r2plus1d_34_clip32_ig65m_from_scratch-449a7af9.pth](https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip32_ig65m_from_scratch-449a7af9.pth)를 다운 받아 

weights/안에 넣었고 utils/model_path.py를 적절히 수정했습니다.

## Dataset Format
In order to implement training and validation the list of training and validation samples should be created as txt files in \datasets\settings folder. 

As an example, the settings for hmdb51 is added. In the file names train-val, modality and split of the dataset is specified. In the txtx files, the folder of images, the number of frames in the folder and the index of the class is denoted, respectively.

The image folders should be created in \datasets as and hmdb51_frames and ucf101_frames for hmdb and ucf datasets. If you want use this code in separate dataset, there is a need to create .py file like hmdb51.py and ucf101.py existing in \dataset. You can just copy these py. files and change the class name of the dataset. Then the init file in the dataset should be modified properly.  

The format of the rgb and flow images is determined for hmdb51 as

"img_%05d.jpg", "flow_x_%05d", "flow_y_%05d" 

and for ucf101 as 

"img_%05d.jpg", "flow_x_%05d.jpg", "flow_y_%05d.jpg" 

Name patterns of the dataset can be modified but the test of the datasets should also be modified. These are also specified in the variable called extension in the test files.

위의 설명을 도식으로 표현하면 아래와 같습니다.

cvpr_frames의 하위 폴더는 제가 공유해드린 [tv-l1 repo](https://github.com/bomtorazek/py-denseflow)
의 실행결과인 flows 폴더를 복붙하시면 됩니다.

```
datasets	
	cvpr_frames
		0 (validation frames)
			img_00001.jpg
			flow_x_00001.jpg
			flow_y_00001.jpg
		1
		2
		...
		Run1_1 (training frames)
		...
	settings
		cvpr
			train_rgb_split0.txt
			val_rgb_split0.txt
			test_rgb_split0.txt
```
#### settings 만들기
설명 드렸던 것과 같이 cvpr_frames에 프레임을 다 넣어놓고

저희의 imageset처럼 settings 안에서 데이터를 나누는 방식입니다.

datasets/settings/make_settings.py를 실행하시면 됩니다. (맨 위의 변수들 (trainvaltest부터 test_lbl_path까지) 을 적절히 수정해주세요.)

최상위 폴더에 있는 ARID1.1_t1_validation_gt_pub.csv 파일은 레이블링 해주신 것을 참고하여 만들었습니다.

아직 5-fold, semi(Track 2.2) 기능은 추가하지 않았습니다.

## Training of the dataset
There are two seperate training files called two_stream2.py and two_stream_bert2.py. These are almost identical two training files. Select the first for SGD training and select the second for ADAMW trainings. 

#### Models
For the models listed below, use two_stream2.py

- **rgb_resneXt3D64f101**
- **flow_resneXt3D64f101**
- **rgb_slowfast64f_50**
- **rgb_I3D64f**
- **flow_I3D64f**
- **rgb_r2plus1d_32f_34**

For the models listed below, use two_stream_bert2.py
- **rgb_resneXt3D64f101_bert10_FRAB**
- **flow_resneXt3D64f101_bert10_FRAB**
- **rgb_resneXt3D64f101_bert10_FRMB**
- **flow_resneXt3D64f101_bert10_FRMB**
- **rgb_resneXt3D64f101_FRMB_adamw**
- **rgb_resneXt3D64f101_adamw**
- **rgb_resneXt3D64f101_FRMB_NLB_concatenation**
- **rgb_resneXt3D64f101_FRMB_lstm**
- **rgb_resneXt3D64f101_concatenation**

- **rgb_slowfast64f_50_bert10_FRAB_late**
- **rgb_slowfast64f_50_bert10_FRAB_early**
- **rgb_slowfast64f_50_bert10_FRMB_early**
- **rgb_slowfast64f_50_bert10_FRMB_late**

- **rgb_I3D64f_bert2**
- **flow_I3D64f_bert2**
- **rgb_I3D64f_bert2_FRMB**
- **flow_I3D64f_bert2_FRMB**
- **rgb_I3D64f_bert2_FRAB**
- **flow_I3D64f_bert2_FRAB**

- **rgb_r2plus1d_32f_34_bert10**
- **rgb_r2plus1d_64f_34_bert10**

#### Training Commands

```

python two_stream_bert2.py --split=1 --arch=rgb_resneXt3D64f101_bert10_FRMB --workers=2 --batch-size=8 --iter-size=16 --print-freq=400 --dataset=hmdb51 --lr=1e-5

python two_stream2.py --split=1 --arch=rgb_resneXt3D64f101 --workers=2 --batch-size=8 --iter-size=16 --print-freq=400 --dataset=hmdb51 --lr=1e-2

제가 사용한 command는 다음과 같습니다.
python two_stream_bert2.py --split=0 --arch=rgb_r2plus1d_32f_34_bert10 --workers=2 --batch-size=4 --iter-size=16 --print-freq=400 --dataset=cvpr --lr=1e-5

```
For multi-gpu training, comment the two lines below in two_stream_bert2.py

혹시 성능이 떨어질까 우려되어 (사용해본 적이 없네요.) multi-는 안 해봤는데 경험있으신 분 말씀드립니다.
```
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

```
To continue the training from the best model, add -c. 
To evaluate the single clip single crop performance of best model, add -e (사용할 일이 없을 것 같습니다.)

## Test of the dataset

이 기능은 clip을 사용하는 것 같아 사용하지 않았습니다.

대신 two_stream_bert2_inference.py 를 사용하시면 accuracy, track1_pred.csv (제출용), track1_prob.csv (앙상블용)를 얻을 수 있습니다.

이 코드의 경우 settings 안의 split.txt에 gt가 반드시 필요한데요.

gt 가 없는 testset의 경우 (4/28 이후) test_rgb_split0.txt를 만들 시 gt class 부분을 아무렇게나 넣어서 만들면 됩니다.

제가 사용한 command는 다음과 같습니다.
```
python two_stream_bert2_inference.py --split=0 --arch=rgb_r2plus1d_32f_34_bert10 --workers=2 --batch-size=4 --iter-size=16 --print-freq=400 --dataset=cvpr
```

For the test of the files, there are three seperate files, namely 
spatial_demo3D.py  -- which is for multiple clips test
spatial_demo_bert.py  -- which is for single clip test
combined_demo.py  -- which is for two-stream tests

Firstly, set your current directory where the test files exists which is:
scripts/eval/

Then enter the example commands below:
python spatial_demo3D.py --arch=rgb_resneXt3D64f101_bert10_FRMB --split=2

python spatial_demo_bert.py --arch=flow_resneXt3D64f101_bert10_FRMB --split=2

python combined_demo.py --arch_rgb=rgb_resneXt3D64f101_bert10_FRMB  --arch_flow=flow_resneXt3D64f101_bert10_FRMB --split=2

If your training is implemented with multi-GPU, manually set multiGPUTrain to True
As default, the tests are implemented ten crops. For single crops test, manually set ten_crop_enabled to False

## Related Projects
[Toward Good Practices](https://github.com/bryanyzhu/two-stream-pytorch): PyTorch implementation of popular two-stream frameworks for video action recognition

[ResNeXt101](https://github.com/kenshohara/video-classification-3d-cnn-pytorch): Video Classification Using 3D ResNet

[SlowFast](https://github.com/facebookresearch/SlowFast): PySlowFast

[R2+1D-IG65](https://github.com/moabitcoin/ig65m-pytorch): IG-65M PyTorch

[I3D](https://github.com/piergiaj/pytorch-i3d): I3D models trained on Kinetics












