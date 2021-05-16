import os
import PIL
import numpy as np
from tkinter import *
from PIL import Image, ImageDraw, ImageGrab
import cv2 as cv

#creating main window and giving it name(~root)
root = Tk()
root.resizable(0,0)
root.title("Hand Digit Recognizer GUI")

#intilizing few variables
lastx, lasty = None, None
image_number = 0

#creating canvas for drawing
cv = Canvas(root, width = 640, height =480, bg='white')
cv.grid(row=0,column=0,pady=2,columnspan=2)

#clear_widget Function
def clear_widget():
    global cv
    cv.delete("all")

#draw_lines function
def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    cv.create_line((lastx,lasty,x,y),width=7,fill='black',
                    capstyle=ROUND,smooth=TRUE,splinesteps=12)
    lastx, lasty = x,y


#activate_event Function defining <B1-Motion>
def activate_event(event):
    global lastx, lasty
    cv.bind('<B1-Motion>',draw_lines)
    lastx, lasty = event.x, event.y

#Recognize_Digit() function
def Recognize_Digit():
    global image_number
    predictions = []
    percentage = []
    #as we know that image_number=0
    filename = f'image_{image_number}.png'
    widget=cv

    #getting the widget coordinates
    x=root.winfo_rootx()+widget.winfo_x()
    y=root.winfo_rooty()+widget.winfo_y()
    x1=x+widget.winfo_width()
    y1=y+widget.winfo_height()
    #grab the image, crop it according to my requirement and saved it in png figure_format
    ImageGrab.grab().crop((x,y,x1,y1)).save(filename)

    #now using cv2
    #read image in color format
    image = cv.imread(filename,1)
    #converting the loaded image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #applyimg otsu thresholding
    ret,th = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    #findcontour() this function helps in extracting the countours in the image
    contours = cv.findContours(th, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)[0]

    for cnt in contours:
        #getting the bounding box and extract Recognize
        x,y,w,h = cv.boundingRect(cnt)
        #create rectangle
        cv.rectangle(image,(x,y),(x+w, y+h),(255,0,0),1)
        top=int(0.05* th.shape[0])
        bottom=top
        left = int(0.05* th.shape[1])
        right = left
        th_up = cv.copyMakeBorder(th, top, bottom, left, right, cv.BORDER_REPLICATE)
        





#Activate event command
cv.bind('<Button-1>', activate_event)

#Now lets add some buttons and labels
btn_save = Button(text='Recognize Digit', command = Recognize_Digit)
btn_save.grid(row=2,column=0,pady=1,padx=1)
button_clear = Button(text='Clear Widget', command = clear_widget)
button_clear.grid(row=2,column=1,pady=1,padx=1)



root.mainloop()
