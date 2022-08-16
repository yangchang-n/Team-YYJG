#!/usr/bin/env python
# coding: utf-8

# In[11]:


####################################################################################################################
##ver002

#001 : score+angle , 같은구도의 사진이 너무 많이 출력됨.
#002 : score, angle도 score화 하여 angle값이 0에 가까울수록 좋은사진이라고 판별하고 저장, 같은구도는 생략하도록 코드수정



####################################################################################################################
##사용모듈

import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
import time
from collections import deque
import tensorflow as tf

####################################################################################################################
##사용함수
def cal_d(p,n):
    d = (p[0]-n[0])**2 + (p[1]-n[1])**2
    return d**(1/2) 

def cal_p(x,y):
    return (x-y)

def get_score(data):
    score = score_model.predict(data)[0][0]
    score = int(score*10000)
    score = float(score)/100
    return score

def get_angle(data):
    p = angle_model.predict(data)[0][0]
    return p-90

def cor_histogram(correl_imagelist):
    hists = []
    co01 = []
    co09 = []
    for file in correl_imagelist:
        nowimg = file[0]
        #// BGR 이미지를 HSV 이미지로 변환
        hsv = cv2.cvtColor(nowimg, cv2.COLOR_BGR2HSV)
        #// 히스토그램 연산(파라미터 순서 : 이미지, 채널, Mask, 크기, 범위)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        #// 정규화(파라미터 순서 : 정규화 전 데이터, 정규화 후 데이터, 시작 범위, 끝 범위, 정규화 알고리즘)
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
       # // hists 리스트에 저장
        hists.append(hist)

    #// 1번째 이미지를 원본으로 지정
    if hists:
        query = hists[-1]

        #// 비교 알고리즘의 이름들을 리스트에 저장
        methods = ['CORREL']#, 'CHISQR', 'INTERSECT', 'BHATTACHARYYA', 'EMD']
        compare_hists = []
        for i, histogram in enumerate(hists):
            ret = cv2.compareHist(query, histogram, 0)
            if ret<0.9:
                co01.append(correl_imagelist[i])
            else:
                co09.append(correl_imagelist[i])

        co09 = sorted(co09, key = lambda x : x[1])


        while len(co09) > correl_save_number:
            del co09[0]

        correl_imagelist = co01 + co09
        correl_imagelist = sorted(correl_imagelist, key=lambda x : x[1])

        while len(correl_imagelist)> picture_save_number:
            del correl_imagelist[0]
            
    return correl_imagelist

####################################################################################################################
#pretrained model
score_model = tf.keras.models.load_model('./flp/model/flp_score_predict001.h5')
angle_model = tf.keras.models.load_model('./flp/model/flp_horizon_correction003.h5')


####################################################################################################################
#main code

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
count=0
BG_COLOR = (0, 0, 0) # gray
MASK_COLOR = (1, 1, 1) # white

cap = cv2.VideoCapture('./test02.mp4')

prev_time = 0
FPS = 10
prescore=0
idx=0
datacompare = 0

picture_save_number = 30
correl_save_number = 5

imagelist = []
correl_imagelist = []

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        idx+=1
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
    # If loading a video, use 'break' instead of 'continue'.
            break

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        
        n=[]
        visibility =[]
        if results.pose_landmarks:
            for data_point in results.pose_landmarks.landmark:
                n.append(data_point.x)
                n.append(data_point.y)
                n.append(data_point.z)
                visibility.append(data_point.visibility)
        else:
            for _ in range(99):
                n.append(0)

        nowdata = [n]
        
        if datacompare ==0:
            predata = [[0 for _ in range(99)]]
        
        datacompare +=1
        
        #프레임간 좌표이동값
        xyzd = 0
        for i in range(99):
            xyzd += (nowdata[0][i]-predata[0][i])**2

        lifescore = get_score(nowdata)
        lifeangle = get_angle(nowdata)
        
#         allscore = (lifescore+(100-abs(lifeangle)))/2
        allscore = lifescore
        predata = nowdata
        
        text1 = "score : {}".format(round(lifescore,2))
        text2 = "angle : {}".format(round(lifeangle,2))
        org1 = (30,30)
        org2 = (30,60)
        font=cv2.FONT_HERSHEY_SIMPLEX
 
    # Draw landmark annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        save_image = image.copy()

        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        #이상한사진 감지 후 image추가
        if xyzd<0.002 and abs(lifeangle)<10:
            correl_imagelist.append([save_image, allscore])

        correl_imagelist = cor_histogram(correl_imagelist)
            
        prescore= lifescore
        cv2.putText(image, text1, org1, font, 1, (255,0,0) ,2)
        cv2.putText(image, text2, org2, font, 1, (255,0,0) ,2)
        
    # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Holistic', image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        
cv2.destroyAllWindows()
cap.release()


##bestcut 저장
count2=0
for bestpicture in correl_imagelist:
    count2 += 1
    cv2.imwrite('./flp/test002/best_image{:0>4}_s{}'.format(count2, int(bestpicture[1]))+'.png', bestpicture[0])

