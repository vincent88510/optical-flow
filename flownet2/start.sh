#!/bin/bash

rm frames/*.png

#python3 test.py

python3 script_config_egg.py

python3 main.py --inference --model FlowNet2 --save_flow --save ./output --inference_dataset ImagesFromFolder --inference_dataset_root ./frames/ --resume ./FlowNet2_checkpoint.pth.tar

python3 frames_to_video.py