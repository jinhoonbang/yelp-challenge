import cv2
import numpy as np
import os
import glob
from scipy.cluster.vq import kmeans, vq

k = 100

curr_dir = os.path.dirname(os.path.realpath(__file__))
path_photos = os.path.join(curr_dir, '/'.join(("data","photos","*.jpg")))

sift = cv2.xfeatures2d.SIFT_create()
n_row = len(glob.glob(path_photos))
n_row = 20
des_list = []

counter = 0

for image_path in glob.glob(path_photos):
    counter += 1
    if counter > n_row:
        break
    im = cv2.imread(image_path)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    kpts, des = sift.detectAndCompute(gray, None)
    des_list.append((image_path, des))



descriptors = des_list[0][1]
print(descriptors)
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))
    print(descriptors)

print(descriptors.shape)

voc, variance = kmeans(descriptors, k, 1)

im_features = np.zeros((n_row, k), "float32")
for i in range(n_row):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w] += 1

print(im_features)
print(im_features.shape)




#temp = "/Users/jinbang-mac/development/yelp-challenge/data/photos/_bSZkEjJCxK37dFXvjowjw.jpg"
#img = cv2.imread(temp)
# gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# sift = cv2.xfeatures2d.SIFT_create()
# kp = sift.detect(gray,None)

# img=cv2.drawKeypoints(gray,kp,img)
# cv2.imwrite('sift_keypoints1.jpg',img)
# img=cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imwrite('sift_keypoints2.jpg',img)
