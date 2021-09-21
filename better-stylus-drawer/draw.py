import cv2
import pynput
import numpy as np

#main thing
shift_pressed=False
class track_shift:
    def __init__(self):
        self.shift_pressed=False
    def is_pressed(self):
        self.shift_pressed=True
    def is_released(self):
        self.shift_pressed=False
shift_pressed=track_shift()
global_color=(255,255,255)
global_thiccness=2
screen=np.zeros((512,512,3),dtype=np.uint8)
def mouse_event(event,x,y,flags,param):
    global mouseX,mouseY
    if shift_pressed.shift_pressed:
        cv2.circle(screen,(x,y),global_thiccness,global_color,-1)
        mouseX,mouseY=x,y
cv2.namedWindow('screen')
cv2.setMouseCallback('screen',mouse_event)
def nothing(*args,**kwargs):
    return

#cv2.createTrackbar('R','screen',0,255,nothing)
#cv2.createTrackbar('G','screen',0,255,nothing)
#cv2.createTrackbar('B','screen',0,255,nothing)
#cv2.createTrackbar('thiccness','screen',1,10,nothing)
tabs={}
tab_nums={0:id(screen)}
tab_count=0
listener=pynput.keyboard.Listener(on_press=lambda *x: shift_pressed.is_pressed(),on_release=lambda *x:shift_pressed.is_released())
listener.start()
nums=[ord(str(i)) for i in range(10)]
try:
    while 1:
  #      r=cv2.getTrackbarPos('R','screen')
 #       g = cv2.getTrackbarPos('G','screen')
#        b = cv2.getTrackbarPos('B','screen')
   #     global_thiccness=cv2.getTrackbarPos('thiccness','screen')
    #    global_color=(r,g,b)

        cv2.imshow('screen',screen)
        k=cv2.waitKey(1)&0xFF
        
        if k==ord('q'):
            break
        elif k==ord('e'):
            if global_color==(255,255,255):
                global_color=(0,0,0)
            elif global_color==(0,0,0):
                global_color=(255,255,255)
finally:
    print('finally')
    listener.stop()
print('done')
