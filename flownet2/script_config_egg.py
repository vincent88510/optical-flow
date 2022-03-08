
import os 
os.sys.path.append('/home/ibrahima/.local/lib/python3.8/site-packages/resample2d_cuda-0.0.0-py3.8-linux-x86_64.egg')

os.sys.path.append('/home/ibrahima/.local/lib/python3.8/site-packages/correlation_cuda-0.0.0-py3.8-linux-x86_64.egg')

os.sys.path.append('/home/ibrahima/.local/lib/python3.8/site-packages/channelnorm_cuda-0.0.0-py3.8-linux-x86_64.egg')


import os
def mkdir_ifnotexists(dir):
  if os.path.exists(dir):
    return
  os.mkdir(dir)
 

vid_file = './video.mp4'
frame_pth = './frames'
mkdir_ifnotexists(frame_pth)
cmd = "ffmpeg -i %s -start_number 0 -vsync 0 %s/frame_%%06d.png" % (
          vid_file,
          frame_pth,
        )
os.system(cmd)



