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

First, install the tensorflow library

    pip install tensorflow

<br/>

Download the model on http://bit.ly/tfoptflow and put it in

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

First, install the good version of PyTorch :

    pip3 install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

<br/>

    pip install pypng tensorboardx setproctitle colorama scipy matplotlib opencv-python

Now, you will need to download the checkpoint for the inference, can be find at the link :

<br/>

    https://drive.google.com/u/0/uc?id=1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da&export=download

To make the visualization to work, we need the open source software ffmpeg. You can find it on the Ubuntu app store or download it with RPM on terminal.

<br/>

Put an input video in the flownet2 folder and rename it "video.mp4".

Finally run the following script to process a video and create two resulting videos named "FlowVideo.mkv" and "ArrowVideo.mkv" (you can convert it back to mp4 with ffmpeg, but VLC will do just fine with mkv):

    chmod +x start.sh
    ./start.sh
