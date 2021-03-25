CUDA_VISIBLE_DEVICES=3 python two_stream_bert2.py --split=0 --arch=rgb_r2plus1d_32f_34_bert10 --workers=4 --batch-size=4 --iter-size=16 --print-freq=400 --dataset=cvpr --lr=1e-5 -c
