python two_stream_bert2.py --split=1 --arch=rgb_r2plus1d_64f_34_bert10 --workers=4 --batch-size=2 --iter-size=16 --print-freq=400 --dataset=cvpr --lr=1e-5 --light_enhanced --gpu 1 #--evaluate
#python two_stream_bert2.py --split=1 --arch=rgb_r2plus1d_64f_34_bert10 --workers=4 --batch-size=2 --iter-size=16 --print-freq=400 --dataset=cvpr --lr=1e-5 --gpu 2 --mix_type cutout  #--evaluate

# hchoi
python two_stream_bert2.py --split=51 --arch=flow_resneXt3D64f101_bert10_FRMB --workers=8 --batch-size=8 --iter-size=16 --print-freq=400 --save_dir=/data/hyeokjae/trained/workshop --lr=1e-5 --gpu 1 --dataset=cvpr
python two_stream_bert2.py --split=51 --arch=flow_resneXt3D64f101_bert10_FRMB --workers=8 --batch-size=8 --iter-size=16 --print-freq=400 --save_dir=/data/hyeokjae/trained/workshop --lr=1e-5 --gpu 2 --dataset=cvpr_sid
python two_stream_bert2.py --split=51 --arch=flow_resneXt3D64f101_bert10_FRMB --workers=8 --batch-size=8 --iter-size=16 --print-freq=400 --save_dir=/data/hyeokjae/trained/workshop --lr=1e-5 --gpu 3 --dataset=cvpr_sid_sm