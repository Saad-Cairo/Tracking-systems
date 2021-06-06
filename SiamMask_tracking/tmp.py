import cv2
import pickle
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import pandas as pd
from Frame_manager import *
from utils import *
from output_saver import *



#output_coord = Output_coordinator('/media/innolab/3448C50948C4CB36/tracking_test/tracking.pkl', '/media/innolab/3448C50948C4CB36/tracking_test', '/media/innolab/3448C50948C4CB36/tracking_test/detection_imgs')
#output_coord.make_output_video_from_pickle()

pkl_path = '/media/innolab/3448C50948C4CB36/tracking_test/tracking.pkl'

#connect_tracks_from_pickle(pkl_path, '/home/khaled/Work/tmp/tracking_test',60, 60)


pkl_file = open(pkl_path, 'rb')
dict = pickle.load(pkl_file)
pkl_file.close()
print(dict)


'''
ID = -1, x1 = 1618, y1 = 604, x2 = 1653, y2 = 765, type = None
[[1618.8884  788.4934]
 [1569.4739  604.3119]
 [1653.8401  581.677 ]
 [1703.2546  765.8585]]
'''


'''box1 = Box((1618, 604, 1653, 765), None)
box2 = Box((1499, 662, 1531, 783), None)


print(box_box_iou(box1, box2))
img = cv2.imread('/media/innolab/3448C50948C4CB36/tested_video_temp/october_bridge/detection_images/F_000001829.jpg')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
p = [[1618.8884, 788.4934], [1569.4739, 604.3119], [1653.8401, 581.677 ], [1703.2546, 765.8585]]
box = convert_polygon_to_box(p)
p = np.array(p)
p = p.reshape((-1, 1, 2))
cv2.polylines(img, np.int32([p]), True ,(255, 0, 0), 3)

print((box.x2, box.y2))
cv2.rectangle(img, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), (0, 255, 0), 2)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#cv2.rectangle(img, (box2.x1, box2.y1), (box2.x2, box2.y2), (0, 255, 0), 2)
#print(box1.box_iou(box2))
plt.imshow(img)
plt.show()'''

'''
img_path = '/home/khaled/Work/tmp/tracking_test/detection_imgs'
csv_path = '/home/khaled/Work/tmp/tracking_test/detections.csv'
fm = FrameManager(csv_path, img_path)
data = fm.read_detection_output()

csv_file = pd.read_csv(csv_path)
i = 250

while i < 290:
    img = cv2.imread(img_path+'/F_'+str(i).zfill(9)+'.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for d in data:
        if d[0] == i:
            cv2.rectangle(img, (d[1], d[2]), (d[3], d[4]), (255, 0, 0), 2)

    plt.imshow(img)
    plt.show()
    i += 1'''



'''img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pts = np.array([[612,574],[775,585], [765,740], [595,740]], np.int32)
print(pts.shape)
pts = pts.reshape((-1,1,2))
print(pts)
print(img.shape)
cv.polylines(img,[pts],True,(0,0,255), 3)
plt.imshow(img)
plt.show()'''
'''file_path = '/home/khaled/Work/tmp/tracking_test/tracking.pkl'
file = open(file_path, 'rb')
dictt = pickle.load(file)
file.close()
print(dictt['AD0'])'''

'''video_path = '/home/khaled/Work/tmp/tracking_test/test_video.MP4'
cap = cv2.VideoCapture(video_path)
imgs_path = '/home/khaled/Work/tmp/tracking_test/detection_imgs'

i = 0
while cap.isOpened():
    ret, frame = cap.read()

    if ret == True:
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cv2.imwrite(imgs_path+'/F_'+str(i).zfill(9)+'.jpg', frame)
        i +=1'''
