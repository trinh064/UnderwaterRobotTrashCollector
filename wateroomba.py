## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import scipy.ndimage
import keyboard
import os
import queue
import tensorflow as tf
from tensorflow import keras
import math
from adafruit_servokit import ServoKit
import board
import busio
import time
import motor

print("Initializing Servos")
i2c_bus1=(busio.I2C(board.SCL, board.SDA))
print("Initializing ServoKit")
kit = ServoKit(channels=16, i2c=i2c_bus1)
print("Done initializing")
for i in range(9):
    motor.down()
model=tf.keras.models.load_model('TrashCNN.h5')
    # org
font = cv2.FONT_HERSHEY_SIMPLEX
org = (20, 30)
    # fontScale
fontScale = 0.25
    # Blue color in BGR
color = (255,255,255)
    # Line thickness of 2 px
thickness = 1
timer=0
maxindx=1
prob=0

#Where the CNN works
def process_img(src):
    imgforCNN=cv2.resize(src,dsize=(160,160))
    length=imgforCNN.shape[0]
    imgforCNN=imgforCNN.reshape((1,length,length,3))
    imgforCNN=np.array(imgforCNN)
    #print(imgbatch.shape)
    predictions=model.predict_on_batch(imgforCNN).flatten()
    result = tf.nn.sigmoid(predictions)
    #print(result)
    #result=1
    return result[0]

def analyze(line):
    p1=np.array([line[0],line[1]])
    p2=np.array([line[2],line[3]])
    dv=p1-p2
    length=np.sqrt(dv[0]**2+dv[1]**2)
    if dv[1]==0:
        if dv[0]>0:
            angle=math.pi/2
        else:
            angle=-math.pi/2
    else:
        angle=np.arctan(dv[0]/dv[1])

    return [length,angle]
def averageAngle(lines):
    llist=[]
    alist=[]
    for i in range(0, len(lines)):
        l = linesP[i][0]
        rs=analyze(l)
        llist+=[rs[0]]
        alist+=[rs[1]]
    #print(list)
    iofmax=np.argmax(llist)
    return -alist[iofmax]*360/(2*math.pi)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
count=0
newsizelist=np.zeros(100)
index=0
angle=90
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        timer=(timer+1)%100
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())[:300,170:470]
        
        color_image = np.asanyarray(color_frame.get_data())[:300,170:470]
        #find the average of center of depth image
        holder=depth_image.astype(np.float32)
        holder[holder==0]=np.nan
        center_mean=np.nanmean(holder[150+30:150+90,150:150+40])
        if np.isnan(center_mean):
             center_mean=1000
        else:
             center_mean=int(center_mean)
        
        depth_mean=int(np.nanmean(holder[0:150,:]))

        #newsize=480
        #if(center_mean>250):
        #    newsize=int((480*(center_mean-30)/250))
        #half_newsize=int(newsize/2)

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.3), cv2.COLORMAP_JET)
      
        x_offset=240-10
        y_offset=240-10
        #rotate servo
        kit.servo[0].angle=angle
        objcenter=depth_mean-center_mean>30
        # Process image every 5 frames
        if timer%5==0:
            result= process_img(color_image)
            resultp=process_img(scipy.ndimage.rotate(color_image,15,reshape=False))
            resultn=process_img(scipy.ndimage.rotate(color_image,-15,reshape=False))
            temp=[resultp,result,resultn]
            maxindx=np.argmax(temp)
            #set angle to one with the highest probability
            prob=temp[maxindx]
            if maxindx==0:
                angle=(angle+10)
                if angle>180:
                  angle=180
            if maxindx==2:
                angle=(angle-10)
                if angle <0:
                  angle= 0

        label=prob>0.9
        
        within=center_mean<=240

        depth_colormap = cv2.rectangle(depth_colormap,(0,0),(300,80),(0,0,0),-1)
        depth_colormap = cv2.putText(depth_colormap, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
        depth_colormap = cv2.putText(depth_colormap,"OBJ IN CENTER:%d - ANGLE:%f - INCLAW:%d"%(objcenter,angle,within&objcenter), (20,60), font, fontScale, color, thickness, cv2.LINE_AA)
        depth_colormap = cv2.resize(depth_colormap, (540, 540))
  
        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape
        print("Center: ",center_mean,"Background: ",depth_mean)
               
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))
        cv2.imshow('window', images)
        if(within&objcenter& label):
            while(1):
             motor.up()
             images = cv2.putText(images, "Closing gripper ... hold space to stop", (20,30), font, 0.75, color, 2, cv2.LINE_AA)
             cv2.imshow('window', images)
             cv2.waitKey(1)
             if cv2.waitKey(33) == ord(' '):
               images = cv2.putText(images, "Resetting linear actuator ...", (20,65), font, 0.75, color, 2, cv2.LINE_AA)
               cv2.imshow('window', images)
               cv2.waitKey(1)
               print("Resetting linear actuator ...")
               for i in range(9):
                 motor.down()
               images = cv2.putText(images, "Done resetting, press space to start", (20,100), font, 0.75, color, 2, cv2.LINE_AA)
               cv2.imshow('window', images)
               cv2.waitKey(1)
               print("Done resetting, press space to start")
               while(1):
                 if cv2.waitKey(33) == ord(' '):
                   break
               break

        if cv2.waitKey(33) == ord('q'):
            exit()
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()
