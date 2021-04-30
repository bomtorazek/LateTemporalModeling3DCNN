import os 
import argparse

parser = argparse.ArgumentParser("Inference")
parser.add_argument('--gpu', default=0, type=int, help='use gpu with cuda number')
parser.add_argument('--model_path', type=str)
args = parser.parse_args()

model_path = args.model_path # "/data_hdd/hoseong/videoexp/cvpr_rgb_r2plus1d_64f_34_bert10_split0_mixtype_None_optimizer_AdamW_randaug_1_3/"

core_cmd = "python two_stream_bert2_inference.py --split=00 --arch=rgb_r2plus1d_64f_34_bert10 --workers=4 --batch-size=2 --dataset=cvpr --gpu {} --tta 1 \
--model-path={}".format(args.gpu, model_path)


ckpt_list = os.listdir(model_path)
ckpt_list = [file for file in ckpt_list if file.endswith(".tar")]
print(ckpt_list)

for ckpt in ckpt_list:
    os.system(core_cmd + ckpt)
