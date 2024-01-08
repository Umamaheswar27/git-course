import cv2

from ctypes import *

img = cv2.imread('test3.jpeg')

classNames = []

classFile = 'coco.names'

with open(classFile,'rt') as f:

    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)

net.setInputSize(320,320)

net.setInputScale(1.0/127.5)

net.setInputMean((127.5, 127.5, 127.5))

net.setInputSwapRB(True)

classIds, confs, bbox = net.detect(img,confThreshold=0.5)

print(classIds,bbox)

count = 0

flag = 0

for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
    
    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
    
    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    
    if classId == 1:
    
        count = count + 1
    
    elif classId == 3:
    
        flag = flag + 1

print("No Of Persons = ",count)

#print("No Of Cars = ",flag)

cv2.imshow("Output",img)

cv2.waitKey(0)