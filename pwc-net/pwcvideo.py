from __future__ import absolute_import, division, print_function
from copy import deepcopy
from skimage.io import imread
from model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_TEST_OPTIONS
#from visualize import display_img_pairs_w_flows
import numpy as np
import os, os.path
import cv2

vid_file = 'video.mp4'
frame_pth = 'frames'
output_frame_pth = 'frames/output_frames'

if not os.path.exists(frame_pth):
	os.mkdir(frame_pth)

if not os.path.exists(output_frame_pth):
	os.mkdir(output_frame_pth)

if not os.path.exists('output'):
	os.mkdir('output')

cmd = "ffmpeg -i '{0}' -start_number 0 -vsync 0 '{1}'/frame_%06d.png".format(
 vid_file,
 frame_pth
)
os.system(cmd)

#count = os.popen("ffprobe -select_streams v -show_streams '{0}' 2>/dev/null | grep nb_frames | sed -e 's/nb_frames=//'".format(vid_file)).read()
count = len([name for name in os.listdir(frame_pth) if os.path.isfile(frame_pth + name)])

# TODO: Set device to use for inference
# Here, we're using a GPU (use '/device:CPU:0' to run inference on the CPU)
gpu_devices = ['/device:CPU:0']
controller = '/device:CPU:0'

# TODO: Set the path to the trained model (make sure you've downloaded it first from http://bit.ly/tfoptflow)
ckpt_path = './models/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000'

# Configure the model for inference, starting with the default options
nn_opts = deepcopy(_DEFAULT_PWCNET_TEST_OPTIONS)
nn_opts['verbose'] = True
nn_opts['ckpt_path'] = ckpt_path
nn_opts['batch_size'] = 1
nn_opts['gpu_devices'] = gpu_devices
nn_opts['controller'] = controller

# We're running the PWC-Net-large model in quarter-resolution mode
# That is, with a 6 level pyramid, and upsampling of level 2 by 4 in each dimension as the final flow prediction
nn_opts['use_dense_cx'] = True
nn_opts['use_res_cx'] = True
nn_opts['pyr_lvls'] = 6
nn_opts['flow_pred_lvl'] = 2

# TODO: The size of the images in this dataset are not multiples of 64, while the model generates flows padded to multiples
# of 64. Hence, we need to crop the predicted flows to their original size
nn_opts['adapt_info'] = (1, 360, 640, 2)

# Instantiate the model in inference mode and display the model configuration
nn = ModelPWCNet(mode='test', options=nn_opts)
#nn.print_config()

# Build an image pair to process
img_pair = []
for n in range(718):#int(count)-2):
	image_path1 = f'./frames/frame_{n:06d}.png'
	image_path2 = f'./frames/frame_{n+1:06d}.png'
	img_pair.clear()
	img_pair.append((imread(image_path1), imread(image_path2)))

	# Generate the predictions
	flow = nn.predict_from_img_pairs(img_pair, batch_size=1, verbose=False)
	flow = np.array(flow)
	flow[np.abs(flow) < 1.0] = 0.

	hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.uint8)
	mag, ang = cv2.cartToPolar(flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32))

	# A couple times, we've gotten NaNs out of the above...
	nans = np.isnan(mag)
	if np.any(nans):
		nans = np.where(nans)
		mag[nans] = 0.

	# Normalize
	hsv[..., 0] = ang * 180 / np.pi / 2
	hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
	hsv[..., 2] = 255

	# Convert to RGB and write in a file
	img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
	cv2.imwrite(output_frame_pth + '/frame' + str(n).zfill(6) + '.png', img)
	print("frame" + str(n).zfill(6))

# Create a video
os.system("ffmpeg -r 24 -i " + output_frame_pth + "/frame%6d.png -vcodec libvpx -b 10M -y output/pwc_result_video.mkv")
