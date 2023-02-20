import cv2
import numpy as np
import pyautogui
from pynput.mouse import Button, Controller
import time
import mediapipe as mp

cap = cv2.VideoCapture(0)
time.sleep(1.1) # to let camera adjust to lighting conditions
_, img = cap.read()
mouse = Controller()  #intialise  mouse api
gamma = 0.5
check = False
pts = [(0, 0), (0, 0), (0, 0), (0, 0)]
pointIndex = 0
AR = (1080, 1920) #resolution LOWER == BETTER . Will try in 1080 x 720 tis is height x width
oppts = np.float32([[0, 0], [AR[1], 0], [0, AR[0]], [AR[1], AR[0]]])
a = 0
b = 0
lower = (0, 65, 200)
upper = (90, 175, 255)# define range for lower and upper HSV bounds for the red LED (90, 175, 255), (0,100,100)


def adjust_gamma(image, gamma):

    invGamma = 1.0 / gamma
    table = np.array([((i/255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)

def draw_circle(event ,x ,y,flags, param):
    global img
    global pointIndex
    global pts

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x,y),5,(0,255,0),-1)
        # if len(event) > 2:
        #     cv2.line(img,lines[-1],lines[-2],(255, 0, 255), 5 )
        pts[pointIndex] = (x, y)
        pointIndex = pointIndex + 1
        print(pts)


def show_window():
    while True:
        cv2.imshow('img', img)
        if(pointIndex == 4):
            break
        if(cv2.waitKey(20) & 0xFF == 27):
            break

def get_persp(image,pts):
    ippts = np.float32(pts)
    Map = cv2.getPerspectiveTransform(ippts, oppts) #screen matrices
    warped = cv2.warpPerspective(image, Map, (AR[1], AR[0])) # persp tran algo -- research more tldr: uses Matrices to warp image
    return warped


cv2.namedWindow('img')
cv2.setMouseCallback('img', draw_circle)
print('Top left, Top right, Bottom Right, Bottom left')

show_window()




while True:
    _, frame = cap.read()
    warped = get_persp(frame, pts) # transform

    blurred = cv2.GaussianBlur(warped, (9, 9), 0)#9 9 0 Lessen : add blur it for thresholding
    # adjusted = adjust_gamma(blurred, gamma)
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # adjusted = adjust_gamma(hsv, gamma)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower, upper)



    ret, otsu = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # adaptive ? cv2.THRESH_BINARY + cv2.THRESH_OTSU

    cnts = cv2.findContours(otsu.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2] # instead of nOne which takes 734 points, simple for 4 points
    center = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        a = x
        b = y
        M = cv2.moments(c)

        if M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        else:
            center = (0, 0)

        if(radius > 1): # proceed if radius is bigger than 1
            check = True
            print(radius)
            cv2.circle(frame,(int(x),int(y)), int(radius), (0,255,255),2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)


#mouse - debug reso
    width, height = pyautogui.size()

    m = (a/1920)*100
    n = (b/1080)*100

    k = (width*m)/100
    c = (height*n)/100

    if check == True :
        mouse.position = (int(k), int(c))
    #     mouse.press(Button.left)
    #
    # else:
    #     mouse.release(Button.left)

    check = False


# debug/adjust  here
    cv2.imshow('frame', frame)
    cv2.imshow('dilate', otsu)
    # cv2.imshow('Mask', mask)
    # cv2.imshow('Blurred',blurred)
    # cv2.imshow('HSV',hsv)
    cv2.imshow('Another One', warped)
    k = cv2.waitKey(5) & 0xFF

    if k == 27:
        break
        
    # if cv2.waitKey(1) & 0xFF == ord("q"):


cv2.destroyAllWindows()
    #     break
cap.release()