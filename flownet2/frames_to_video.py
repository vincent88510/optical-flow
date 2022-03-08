import os
import cv2
import PIL.Image
from flow_utils import read_flow, flow_to_image, threshold 
from visualization import draw_arrows


def mkdir_ifnotexists(dir):
  if os.path.exists(dir):
    return
  os.mkdir(dir)


flo_pth = 'output/inference/run.epoch-0-flow-field/'
frame_pth = 'frames/'
flos = [flo_pth + f for f in os.listdir(flo_pth)]
flos.sort()
frms = [frame_pth + f for f in os.listdir(frame_pth)]
frms.sort()
mkdir_ifnotexists('./FlowFrames')
mkdir_ifnotexists('./ArrowFrames')

for i in range(len(flos)):
  flow = read_flow(flos[i])
  img = cv2.imread(frms[i])
  flow_img = flow_to_image(threshold(flow, 1.0))
  draw_arrows(flow, flow_img, img)
  
  PIL.Image.fromarray(flow_img).save('./FlowFrames/'+os.path.basename(flos[i])+'.png')
  PIL.Image.fromarray(img).save('./ArrowFrames/'+os.path.basename(frms[i]))
  

os.system('ffmpeg -r 24 -i FlowFrames/%6d.flo.png -vcodec libvpx -b 10M -y FlowVideo.mkv')
os.system('ffmpeg -r 24 -i ArrowFrames/frame_%6d.png -vcodec libvpx -b 10M -y ArrowVideo.mkv')
