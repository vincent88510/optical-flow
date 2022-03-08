from __future__ import absolute_import, division, print_function
from copy import deepcopy
from model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_TEST_OPTIONS
#from visualize import display_img_pairs_w_flows
import numpy as np
import os, os.path
import cv2
import time

# TODO: Set device to use for inference
# Here, we're using a CPU (use '/device:GPU:0' to run inference on the GPU)
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



cap = cv2.VideoCapture(0)

# Build an image pair to process
img_pair = []
ret, frame = cap.read()
img_pair.append((frame, frame))
listframe = list((frame, frame))

t = 0
dt = 0
while(True):
	ret, frame = cap.read()
	listframe.pop(0)
	listframe.append(frame)
	img_pair.clear()
	img_pair.append(tuple(listframe))

	# Generate the predictions
	flow = nn.predict_from_img_pairs(img_pair, batch_size=1, verbose=False)
	flow = np.array(flow)
	# Noise threshold
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

	# Convert to RGB and show
	img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
	cv2.imshow('Real time', img)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	# Framerate
	dt = time.time() - t
	t = time.time()
	print(str(1 / dt) + "fps")

cap.release()
cv2.destroyAllWindows()
