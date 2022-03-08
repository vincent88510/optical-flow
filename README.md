# optical-flow
Optical flow estimation via different algorithms : Lucas-Kanade, Farneback, FlowNet2, PWC-Net and RAFT.


The very first thing to do is to clone this git repository:

    git clone https://github.com/vincent88510/optical-flow


## Lucas-Kanade
Put an input video in the lucas-kanade folder and rename it "video.mp4".

Run the following script to process a video and showing it frame by frame in a window:

    python3 lucas_kanade.py

<br/>
Press "e" to stop the program.
<br/>
<br/>


## Farneback
Put an input video in the farneback folder and rename it "video.mp4".

Run the following script to process a video and show the result frame by frame in a window:

    python3 farneback.py

<br/>

Run the following script to process a video and create a resulting video named "FarnebackVideo.mkv" (you can convert it back to mp4 with ffmpeg, but VLC will do just fine with mkv):

    python3 farneback_video.py

<br/>
Press "e" to stop the program.
<br/>
<br/>


## pwc-net
Source : https://github.com/philferriere/tfoptflow

<br/>

First, download the model on http://bit.ly/tfoptflow and put it in

    models/pwcnet-lg-6-2-multisteps-chairsthingsmix/

<br/>

Put an input video in the pwc-net folder and rename it "video.mp4".

Run the following script to process a video and create a resulting video named "output/pwc_result_video.mkv" (you can convert it back to mp4 with ffmpeg, but VLC will do just fine with mkv):

    python3 pwcvideo.py

<br/>

Run the following script to process a video and show the result frame by frame in a window:

    python3 pwcrt.py

<br/>
Press "q" to stop the program.
<br/>
<br/>


## flownet2
Source : https://github.com/NVIDIA/flownet2-pytorch

<br/>

Put an input video in the flownet2 folder and rename it "video.mp4".
<br/>
<br/>


## raft
Source : https://github.com/princeton-vl/RAFT

<br/>

Put an input video in the raft folder and rename it "video.mp4".








## Installation 

    # get flownet2-pytorch source
    git clone https://github.com/NVIDIA/flownet2-pytorch.git
    cd flownet2-pytorch

    # install custom layers
    bash install.sh
    
### Python requirements 
Currently, the code supports python 3
* numpy 
* PyTorch ( == 0.4.1, for <= 0.4.0 see branch [python36-PyTorch0.4](https://github.com/NVIDIA/flownet2-pytorch/tree/python36-PyTorch0.4))
* scipy 
* scikit-image
* tensorboardX
* colorama, tqdm, setproctitle 

## Inference
    # Example on MPISintel Clean   
    python main.py --inference --model FlowNet2 --save_flow --inference_dataset MpiSintelClean \
    --inference_dataset_root /path/to/mpi-sintel/clean/dataset \
    --resume /path/to/checkpoints
