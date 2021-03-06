import cv2
import numpy as np
import os

capture = cv2.VideoCapture("video.mp4")

count = 0
while 1:
	# grab the current frame
	(grabbed, frame) = capture.read()
	if not grabbed:
		break
	count += 1

capture = cv2.VideoCapture("video.mp4")

if not os.path.exists('./Frames'):
    os.mkdir('./Frames')

# Reading the first frame
_, frame1 = capture.read()
# Convert to gray scale
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
# Create mask
hsv_mask = np.zeros_like(frame1)
# Make image saturation to a maximum value
hsv_mask[..., 1] = 255

# Till you scan the video
for n in range(1, count):
    # Capture another frame and convert to gray scale
    _, frame2 = capture.read()
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Optical flow is now calculated
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow[abs(flow) < 1.0] = 0
    # Compute magnitude and angle of 2D vector
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # Set image hue value according to the angle of optical flow
    hsv_mask[..., 0] = ang * 180 / np.pi / 2
    # Set value as per the normalized magnitude of optical flow
    hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # Convert to rgb
    rgb_representation = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)

    kk = cv2.waitKey(20) & 0xff
    # Press 'e' to exit the video
    if kk == ord('e'):
        break

    cv2.imwrite('./Frames/frame'+str(n).zfill(6)+'.png', rgb_representation)

    prvs = next

os.system('ffmpeg -r 24 -i Frames/frame%6d.png -vcodec libvpx -b 10M -y FarnebackVideo.mkv')

capture.release()
