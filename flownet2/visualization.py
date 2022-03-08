import numpy as np
import matplotlib.pyplot as plt
import cv2

def draw_arrows(flow, flow_img, img):
  gray = cv2.cvtColor(flow_img, cv2.COLOR_RGB2GRAY)
  (ret, binary) = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

  #plt.axis("off")
  #plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
  #plt.show()

  #plt.axis("off")
  #plt.imshow(cv2.cvtColor(binary, cv2.COLOR_BGR2RGB))
  #plt.show()

  # find contours in the binary image
  (contours, hierarchy) = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_TC89_L1)

  (cX, cY) = (0, 0)
  (r, g, b) = (0, 0, 0)

  for c in contours:
    # calculate moments for each contour
    M = cv2.moments(c)

    # calculate x,y coordinate of center
    if M["m00"] != 0:
      cX = int(M["m10"] / M["m00"])
      cY = int(M["m01"] / M["m00"])
    else:
      (cX, cY) = (0, 0)
    
    u = flow[cY, cX, 0]
    v = flow[cY, cX, 1]

    r = int(flow_img[cY, cX, 0])
    g = int(flow_img[cY, cX, 1])
    b = int(flow_img[cY, cX, 2])
    
    ang = np.angle(complex(u, v))
    mag = 5 * np.abs(complex(u, v))
    mag *= 2
    end = (int(mag * np.cos(ang)) + cX, 
           int(mag * np.sin(ang)) + cY)

    # threshold to avoid noisy contours
    thresh = 127
    if r < thresh or g < thresh or b < thresh:
      flow_img = cv2.arrowedLine(flow_img, (cX, cY), end, 
                            (255 - r, 255 - g, 255 - b), 9)
      img = cv2.arrowedLine(img, (cX, cY), end, (255, 0, 0), 9)

