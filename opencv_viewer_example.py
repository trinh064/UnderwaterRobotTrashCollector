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
model=tf.keras.models.load_model('newsavemodel.h5')
    # org
font = cv2.FONT_HERSHEY_SIMPLEX
org = (20, 30)
    # fontScale
fontScale = 0.5
    # Blue color in BGR
color = (34,139,34)
    # Line thickness of 2 px
thickness = 2

def process_img(src):
    length=src.shape[0]
    img=src.reshape((1,length,length,3))
    imgbatch=np.array(img)
    #print(imgbatch.shape)
    predictions=model.predict_on_batch(imgbatch).flatten()
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

path = './Depth'
first=input("train or validation t/v?:")
if(first=='t'):
    path+='/train'
else:
    path+='/validation'
second=input("success or failure s/f?:")
if(second=='s'):
    path+='/Success'
else:
    path+='/Failure'
print("path: ",path)
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
icon_img = cv2.imread("icon.jpg")
icon_img1= cv2.resize(icon_img, (20,20))
# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
count=0
newsizelist=np.zeros(100)
index=0

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_image=depth_image[:,80:560]
        color_image=color_image[:,80:560]

        center_mean=int(np.average(depth_image[240-10:240+10,240-10:240+10]))
        depth_mean=int( np.average(depth_image[0:240]) )
        #depth_mean=(np.average(depth_image[0:100])+np.average(depth_image[380:]))/2
        #print ("center cluster:" , center_mean)
        #print ("average depth:", depth_mean )

        newsize=480
        if(center_mean>250):
            newsize=int((480*center_mean/250))
            #newsizelist[index]=newsize
            #index=(index+1)%100
            #depth_image-=abs(center_mean-250)
        #print("center:",center_mean)
        #if (np.std(newsizelist)<250):
        #    continue
        half_newsize=int(newsize/2)
        depth_image = cv2.resize(depth_image, (newsize,newsize))[half_newsize-240:half_newsize+240,half_newsize-240:half_newsize+240]
        
        if(center_mean>270):
            depth_image-=center_mean-270
            depth_image[depth_image<0]=0

        #den_list=[]
        #rot_image=depth_image
        '''
        for i in range(4):
            vertical_band=rot_image[240-140:240+140,240-40:240+40]
            den_list+=[np.average(vertical_band)]
            rot_image= scipy.ndimage.rotate(depth_image,i*30, reshape=False)

        rot_image=depth_image
        for i in range(4):
            vertical_band=rot_image[240-160:240+160,240-80:240+80]
            den_list+=[np.average(vertical_band)]
            rot_image= scipy.ndimage.rotate(depth_image,-i*30, reshape=False)

        iter=den_list.index(min(den_list))
        theta=0
        if iter<4:
            theta=-iter*30
        else:
            theta=(iter-4)*30
        print("angle of rotatation",theta)
        '''
        #depth_image = scipy.ndimage.rotate(depth_image, -45, reshape=False)
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.3), cv2.COLORMAP_JET)
        edges = cv2.Canny(depth_colormap[240-100:240+100,240-100:240+100],100,200)
        ret,sob= cv2.threshold(edges,80,255,cv2.THRESH_BINARY)
        linesP = cv2.HoughLinesP(sob, 1, np.pi / 180, 50, None, 50, 10)
        if linesP is not None:
            for i in range(0, len(linesP)):
                #l = linesP[i][0]
                #cv2.line(depth_colormap, (l[0]+140, l[1]+140), (l[2]+140, l[3]+140), (155,100,55), 3, cv2.LINE_AA)
                angle=averageAngle(linesP)
        else:
            angle=0
        depth_colormap= scipy.ndimage.rotate(depth_colormap,angle, reshape=False)


        # Show images
        x_offset=240-10
        y_offset=240-10

        #cv2.putText(images, "my_text", (x_offset+icon_img1.shape[1], y_offset+icon_img1.shape[0]), font, 1.0, (255, 255, 255), 0)

        imgforCNN=cv2.resize(depth_colormap,dsize=(160,160))
        #print(imgforCNN.shape)
        #imgforCNN=screen
        #print(screen.shape)
        result= process_img(imgforCNN)
        label=result>0.5
        objcenter=depth_mean-center_mean>10
        within=center_mean<=350
        text="GUESSVALUE:%f - GRASPABLE:%d"%(result,label)
        depth_colormap = cv2.putText(depth_colormap, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
        depth_colormap = cv2.putText(depth_colormap,"OBJ IN CENTER:%d - ANGLE:%f - INCLAW:%d"%(objcenter,angle,within&objcenter), (20,60), font, fontScale, color, thickness, cv2.LINE_AA)
        depth_colormap = cv2.resize(depth_colormap, (540, 540))
        color_image[y_offset:y_offset+icon_img1.shape[0], x_offset:x_offset+icon_img1.shape[1]] = icon_img1
        # If depth and color resolutions are different, resize color image to match depth image for display
        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))
        cv2.imshow('window', images)

        if cv2.waitKey(33) == ord('s'):
            filename="%d.png"%count
            cv2.imwrite(os.path.join(path,filename), depth_colormap)
            count+=1
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()
