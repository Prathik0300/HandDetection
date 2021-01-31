import cv2 as cv
import numpy as np
from Hand import Hand
import time

def capture_histogram():
    video = cv.VideoCapture(0)
    while True:
        _,frame = video.read()
        frame = cv.flip(frame,1)

        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(frame,"Place your hand in the box and press `a`",(5,50),font,1,(255,255,255),2,cv.LINE_AA)
        cv.rectangle(frame,(400,100),(580,300),(0,255,0),2)

        roi = frame[105:175,505:575]
        cv.imshow("Histogram",frame)
        if cv.waitKey(10) & 0xFF==ord('a'):
            roi_color = roi
            cv.destroyAllWindows()
            break
        
    hsv = cv.cvtColor(roi_color,cv.COLOR_BGR2HSV)
    hist = cv.calcHist([hsv],[0,1],None,[12,15],[0,180,0,256])

    cv.normalize(hist,hist,0,255,cv.NORM_MINMAX)
    video.release()
    return hist

def locate_object(frame,hist):
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    segment = cv.calcBackProject([hsv],[0,1],hist,[0,180,0,256],1)
    disc = cv.getStructuringElement(cv.MORPH_ELLIPSE,(9,9))
    cv.filter2D(segment,-1,disc)
    _,thresh = cv.threshold(segment,70,255,cv.THRESH_BINARY)

    kernel = None
    erode = cv.erode(thresh,kernel,iterations=2)
    dilate= cv.dilate(erode,kernel,iterations=2)
    closing = cv.morphologyEx(dilate,cv.MORPH_CLOSE,kernel,iterations=2)

    mask = cv.bitwise_and(frame,frame,mask=closing)
    return closing,mask,thresh

def detect_hand(frame,hist):
    detected,mask,thresh = locate_object(frame,hist)
    return Hand(detected,mask,thresh,frame)
    

