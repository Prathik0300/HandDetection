import cv2 as cv
import numpy as np
import gesture_recognition as gr


if __name__ == '__main__':
    video = cv.VideoCapture(0)
    hist = gr.capture_histogram()
    
    while True:
        ret,frame = video.read()
        if not ret:
            break
        frame = cv.flip(frame,1)
        gr.detect_face(frame,block=True)
        hand = gr.detect_hand(frame,hist)
        temp = hand.drawOutline()
        outline = hand.drawOutline()
        quick_outline = hand.outline
        fingertipList = hand.findFingertip()
        for fingertip in fingertipList:
            cv.circle(quick_outline,fingertip,5,(0,0,255),-1)
        com = hand.COM()
        if com:
            cv.circle(quick_outline,com,10,(255,0,0),-1)
        cv.imshow("HAND",quick_outline)
        if cv.waitKey(20) & 0xFF==ord('q'):
            break
    video.release()
    cv.destroyAllWindows()
