import cv2 
import numpy as np


def region_of_intrest(image,vertices):
    mask = np.zeros_like(image)
    match_mask_color = 255
    cv2.fillPoly(mask,vertices,match_mask_color)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image

def draw_line(image,lines):
    copy_image = image.copy()
    blank_image = np.zeros((copy_image.shape[0],copy_image.shape[1],3),np.uint8)
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(blank_image,(x1,y1),(x2,y2),(0,255,0),3)
    line_image = cv2.addWeighted(image,0.8,blank_image,1,0)
    return line_image


def process(image):
    height = image.shape[0]
    width = image.shape[1]

    region_of_intrest_vertices = [
        (0,height),
        (width/2,height/2),
        (width,height)
    ]

    gray_image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image,100,200)

    croped_image = region_of_intrest(canny_image,np.array([region_of_intrest_vertices],np.int32))

    lines = cv2.HoughLinesP(croped_image,
                           rho= 6,
                           theta = np.pi/60,
                           threshold =30,
                           lines = np.array([]),
                           minLineLength = 40,
                           maxLineGap=25)

    image = draw_line(image,lines)
    return image


cap = cv2.VideoCapture('test21.mp4')

while(cap.isOpened()):
    ret,frame = cap.read()
    frame = process(frame)
   
    cv2.imshow('analysis video',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()