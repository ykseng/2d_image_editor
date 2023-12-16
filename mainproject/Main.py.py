from tkinter import filedialog,simpledialog
from tkinter import *
from tkinter.colorchooser import askcolor
from turtle import width
import os
import tkinter
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageTk,Image, ImageDraw,ImageGrab


class Paint:
    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'

    def __init__(self):
        self.root = Tk()
        self.root.title("Paint")
        self.root.state("zoom")
        Topframe = Frame(self.root)
        Topframe.pack(fill='x')
        Middleframe = Frame(self.root)  
        Middleframe.pack()
        Bottomframe = Frame(self.root) 
        Bottomframe.pack()

        menubar = Menu(self.root)
        self.root.config(menu = menubar)
        submenu = Menu(menubar,tearoff=0)

        menubar.add_cascade(label='File', menu=submenu)
        submenu.add_command(label='New Canvas', command=self.new)
        submenu.add_command(label='Save as', command=self.save_file)
        submenu.add_command(label='Save',command=self.save)
        submenu.add_separator()
        submenu.add_command(label='Exit',command=self.close)
        tool_menu = Menu(menubar,tearoff=0)
        menubar.add_cascade(label="Tools",menu=tool_menu)
        sub_menu = Menu(tool_menu, tearoff=0)
        sub_menu.add_command(label='Grey Scale Histogram',command=self.greyhistogram)
        sub_menu.add_command(label='Color Histogram',command=self.colorhistogram)
        tool_menu.add_cascade(label='Histogram',menu=sub_menu)
        sub2_menu = Menu(tool_menu, tearoff=0)
        sub2_menu.add_command(label='Gaussion low pass filter',command=self.GaussianimageEnhancement)
        sub2_menu.add_command(label='Average low pass filter',command=self.averageimageEnhancement)
        sub2_menu.add_command(label='high pass filter',command=self.ownimageEnhancement)
        tool_menu.add_cascade(label='Enhance Image',menu=sub2_menu)
        tool_menu.add_command(label='Bit-plane slicing',command=self.bitSlicing)
        sub3_menu = Menu(tool_menu, tearoff=0)
        sub3_menu.add_command(label='Canny Edge Detection',command=self.cannyedgedetection)
        sub3_menu.add_command(label='Prewitt Edge Detection',command=self.prewittedgedetection)
        sub3_menu.add_command(label='Sobel edge detection',command=self.sobeledgedetection)
        tool_menu.add_cascade(label='Edge Detection',menu=sub3_menu)
        tool_menu.add_command(label='contour',command=self.contouring)

        photobrush = PhotoImage(file = "mainproject/icon/brush.png")
        photoeraser = PhotoImage(file = "mainproject/icon/eraser.png")
        photoline = PhotoImage(file = "mainproject/icon/line.png")
        photopencil = PhotoImage(file = "mainproject/icon/pencil.png")
        photopolygon=PhotoImage(file="mainproject/icon/polygon.png") 
        photoInsertImg=PhotoImage(file="mainproject/icon/insertimage.png")
        photocolor=PhotoImage(file="mainproject/icon/color.png")
        photoImgDetial=PhotoImage(file="mainproject/icon/image detial.png")
        photorgb=PhotoImage(file="mainproject/icon/rgb.png")
        photorotate=PhotoImage(file="mainproject/icon/rotate.png")
        photocrop=PhotoImage(file="mainproject/icon/crop.png")
        phototranslation=PhotoImage(file="mainproject/icon/translation.png")
        photoclose=PhotoImage(file="mainproject/icon/close.png")
        photocircle=PhotoImage(file="mainproject/icon/circle.png")
        photoHmerge=PhotoImage(file="mainproject/icon/horizontalmerge.png")
        photoVmerge=PhotoImage(file="mainproject/icon/verticalmerge.png")

        self.pen_button = Button(Topframe, text='Pen',image = photopencil,
                                    command=self.use_pen,height=25,width=25)
        self.pen_button.grid(row=0, column=0)

        self.brush_button = Button(Topframe, text='Brush',image=photobrush,
                                   command=self.use_brush,height=25,width=25)
        self.brush_button.grid(row=0, column=1)

        self.line_button = Button(Topframe, text='Line',image=photoline,
                                  command=self.use_line,height=25,width=25)
        self.line_button.grid(row=0, column=2)

        self.poly_button = Button(Topframe, text='Polygon',image=photopolygon,
                                  command=self.use_poly,height=25,width=25)
        self.poly_button.grid(row=0, column=3)


        self.color_button = Button(Topframe, text='Color',image=photocolor,
                                   command=self.choose_color,height=25,width=25)
        self.color_button.grid(row=0, column=5)

        self.eraser_button = Button(Topframe, text='Eraser',image=photoeraser,
                                    command=self.use_eraser,height=25,width=25)
        self.eraser_button.grid(row=0, column=6)

        self.size_scale = Scale(Topframe, from_=1, to=10,
                                orient='horizontal')
        self.size_scale.grid(row=0, column=7,columnspan=10)

        self.insert_button =Button(Topframe,text="Insert Image",image=photoInsertImg,
                                    command=self.image,height=25,width=25)
        self.insert_button.grid(row=1,column=0)

        self.crop_button = Button(Topframe, text='Crop',image=photocrop,
                                   command=self.cropImage,height=25,width=25)
        self.crop_button.grid(row=1, column=1)

        self.imageDetial_button = Button(Topframe, text='imageDetail',image=photoImgDetial,
                                   command=self.imageDetail,height=25,width=25)
        self.imageDetial_button.grid(row=1, column=2)
        
        self.RGB_button = Button(Topframe, text='RGB',image=photorgb,
                                   command=self.convertRGB,height=25,width=25)
        self.RGB_button.grid(row=1, column=3)

        self.rotateImg_button = Button(Topframe, text='rotateImg',image=photorotate,
                                   command=self.imageRotate,height=25,width=25)
        self.rotateImg_button.grid(row=1, column=4)

        self.translation_button = Button(Topframe, text='translation',image=phototranslation,
                                   command=self.translateImage,height=25,width=25)
        self.translation_button.grid(row=1, column=5)

        self.close_button=Button(Topframe,text='Close',image=photoclose, command=self.close,height=25,width=25)
        self.close_button.grid(row=1,column=6)

        self.Horizonntalmerge_button=Button(Topframe,text='Split and Combine several images',image=photoHmerge, command=self.HseveralImage,height=25,width=25)
        self.Horizonntalmerge_button.grid(row=1,column=7)

        self.Verticalmerge_button=Button(Topframe,text='Split and Combine several images',image=photoVmerge, command=self.VseveralImage,height=25,width=25)
        self.Verticalmerge_button.grid(row=1,column=8)


        self.canvas = Canvas(Middleframe, bg='white', width=1500, height=600)
        self.canvas.grid(row=0, columnspan=10)

        self.var_status = StringVar(value="Selected: Pen")
        self.lbl_status = Label(Bottomframe, textvariable=self.var_status)
        self.lbl_status.grid(row=3, column=4, rowspan=3)

        self.setup()
        self.root.mainloop()

    def new(self):
            self.canvas.delete('all')


    def setup(self):
        self.old_x, self.old_y = None, None
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = None
        self.size_multiplier = 1

        self.activate_button(self.pen_button)
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset)
        self.canvas.bind('<Button-1>', self.point)
        self.root.bind('<Escape>', self.line_reset)
        self.line_start = (None, None)


    def image(self):

        self.image = cv2.cvtColor(cv2.imread(filedialog.askopenfilename()), cv2.COLOR_BGR2RGB)
        self.photo=ImageTk.PhotoImage(image=Image.fromarray(self.image))
        height, width, no_channels = self.image.shape
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

    def HseveralImage(self):
        filez = filedialog.askopenfilenames(parent=self.root, title='Choose a file')
        lst = list(filez)
        imgs    = [ Image.open(i) for i in lst ]
        min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
        imgs_comb = np.hstack([i.resize(min_shape) for i in imgs])
        self.image=(imgs_comb)
        self.photo=ImageTk.PhotoImage(image=Image.fromarray(self.image))
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
    
    def VseveralImage(self):
        filez = filedialog.askopenfilenames(parent=self.root, title='Choose a file')
        lst = list(filez)
        imgs    = [ Image.open(i) for i in lst ]
        min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
        imgs_comb = np.vstack([i.resize(min_shape) for i in imgs])
        self.image=(imgs_comb)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(imgs_comb))
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

    def bitSlicing(self):
        self.gray_image=cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        imgs=[255*((self.gray_image&(1<<i))>>i) for i in range(8)]
        for i in range(8):
            plt.subplot(3,3,i+1)
            plt.title("bit slicing"+ str(i+1))
            plt.imshow(imgs[i],cmap='gray')
            plt.xticks([])
            plt.yticks([])
        plt.show()
    
    def greyhistogram(self):
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        histogram = cv2.calcHist([self.gray_image], [0], None, [256], [0, 256])
        title=['gray image','histogram']
        plt.subplot(2,2,1)
        plt.title(title[0])
        plt.imshow(self.gray_image,cmap='gray')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2,2,2)
        plt.title(title[1])
        plt.plot(histogram, color='k')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def colorhistogram(self):
        for i, col in enumerate(['b', 'g', 'r']):
            hist = cv2.calcHist([self.image], [i], None, [256], [0, 256])
            title=['Image','histogram']
            plt.subplot(2,2,1)
            plt.title(title[0])
            plt.imshow(self.image)
            plt.xticks([])
            plt.yticks([])

            plt.subplot(2,2,2)
            plt.title(title[1])
            plt.plot(hist, color=col)
            plt.xticks([])
            plt.yticks([])
            plt.xlim([0, 256])            
        plt.show()

    def GaussianimageEnhancement(self):
        self.gaussionFilter2=cv2.GaussianBlur(self.image,(7,7),0)
        self.image=(self.gaussionFilter2)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.image))
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

    def averageimageEnhancement(self):
        self.averageFilter2=cv2.blur(self.image,(7,7))
        self.image=(self.averageFilter2)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.image))
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

    def ownimageEnhancement(self):
        kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
        image_sharp = cv2.filter2D(src=self.image, ddepth=-1, kernel=kernel)
        self.ownFilter=cv2.filter2D(self.image,-1,kernel=kernel)
        self.image=(self.ownFilter)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.image))
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

    def cannyedgedetection(self):
        edges = cv2.Canny(self.image,100,200)
        plt.subplot(121),plt.imshow(self.image,cmap = 'gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(edges,cmap = 'gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        plt.show()
    
    def sobeledgedetection(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        img_gaussian = cv2.GaussianBlur(gray,(3,3),0)
        img_sobelx = cv2.Sobel(img_gaussian,cv2.CV_8U,1,0,ksize=5)
        img_sobely = cv2.Sobel(img_gaussian,cv2.CV_8U,0,1,ksize=5)
        img_sobel = img_sobelx + img_sobely
        plt.subplot(121),plt.imshow(self.image,cmap = 'gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(img_sobel,cmap = 'gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        plt.show()

    def prewittedgedetection(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        img_gaussian = cv2.GaussianBlur(gray,(3,3),0)
        kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
        img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
        img_prewitt=img_prewittx + img_prewitty
        plt.subplot(121),plt.imshow(self.image,cmap = 'gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(img_prewitt,cmap = 'gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        plt.show()

    def contouring(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        edged = cv2.Canny(blurred, 10, 100)
        # define a (3, 3) structuring element
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        # apply the dilation operation to the edged image
        dilate = cv2.dilate(edged, kernel, iterations=1)
       
        # find the contours in the edged image
        contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_copy = self.image.copy()
        # draw the contours on a copy of the original image
        cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2)
        print(len(contours), "objects were found in this image.")
        string=str(len(contours))+ " Objects"
        a1=cv2.cvtColor(dilate,cv2.COLOR_GRAY2RGB)
        imgs = [image_copy, a1]
        title = [string,"Dilated image"]
        i = 0
        for i in range (len(imgs)):
            plt. subplot(2,1,i+1)
            plt.title(title[i])
            plt.imshow(imgs[i])
            plt.xticks([])
            plt.yticks([])
        plt.show()

    def imageRotate(self):
        x=simpledialog.askfloat(title="Rotate Image",prompt="degree:")
        scale=1.0
        center = (self.image.shape[1]/2,self.image.shape[0]/2)
        M=cv2.getRotationMatrix2D(center,x,scale)
        rotated=cv2.warpAffine(self.image,M,(self.image.shape[0],self.image.shape[1]))
        self.image=(rotated)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(rotated))
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

    def convertRGB(self):
        
        colorID=simpledialog.askstring(title="Translate Image",prompt="color:")
        
        if colorID=="grey":
            self.grey_image=cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
            self.image=(self.grey_image)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.grey_image))
            self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

        if colorID=="BGRA":
            self.colorImage=cv2.cvtColor(self.image,cv2.COLOR_BGR2BGRA)
            self.image=(self.colorImage)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.colorImage))
            self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

        if colorID=="HSV":
            self.hsv=cv2.cvtColor(self.image,cv2.COLOR_BGR2HSV)
            self.image=(self.hsv)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.hsv))
            self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

        if colorID=="HSV0":
            self.hsv=cv2.cvtColor(self.image,cv2.COLOR_BGR2HSV)
            self.image=(self.hsv[:,:,0])
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.hsv[:,:,0]))
            self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

        if colorID=="HSV1":
            self.hsv=cv2.cvtColor(self.image,cv2.COLOR_BGR2HSV)
            self.image=(self.hsv[:,:,1])
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.hsv[:,:,1]))
            self.canvas.create_image(0, 0, image=self.photo, anchor="nw")  
        
        if colorID=="HSV2":
            self.hsv=cv2.cvtColor(self.image,cv2.COLOR_BGR2HSV)
            self.image=(self.hsv[:,:,2])
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.hsv[:,:,2]))
            self.canvas.create_image(0, 0, image=self.photo, anchor="nw")      
    
    def translateImage(self):
        x=simpledialog.askfloat(title="Translate Image",prompt="x:")
        y=simpledialog.askfloat(title="Translate Image",prompt="y:")
        width,height=self.image.shape[:2]
        translationMatrix=np.float32([[1,0,x],[0,1,y]])
        self.translateImg=cv2.warpAffine(self.image,translationMatrix,(height,width))
        strcaption="Image translated by"+str(x)+","+str(y) 
        self.image=(self.translateImg)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.translateImg))
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

    def imageDetail(self):
        print("image dimension = {}".format(self.image.shape))
        print("image width ={}".format(self.image.shape[1]))
        print("image width ={}".format(self.image.shape[0]))
        print("image no. of channels ={}".format(self.image.shape[2]))
        tkinter.messagebox.showinfo('The image detail',"image dimension = {}".format(self.image.shape)+"\n"+"image width ={}".format(self.image.shape[1])+"\n"+"image width ={}".format(self.image.shape[0])+"\n"+"image no. of channels ={}".format(self.image.shape[2]))

    def cropImage(self):
        a=simpledialog.askfloat(title="Crop Image",prompt="start row:")
        if (a>=1 or a<0):
            tkinter.messagebox.showinfo('Error',"Please insert a integer in the range of 0<=x<1. Example:0.15")
            a=simpledialog.askfloat(title="Rotate Image",prompt="start row:")
        b=simpledialog.askfloat(title="Crop Image",prompt="start column:")
        if (b>=1 or b<0):
            tkinter.messagebox.showinfo('Error','Please insert a integer in the range of 0<=x<1. Example:0.15')
            b=simpledialog.askfloat(title="Rotate Image",prompt="start column:")
        c=simpledialog.askfloat(title="Crop Image",prompt="end row:")
        if (c>1 or c<=0):
            tkinter.messagebox.showinfo('Error','Please insert a integer in the range of 0<x<=1. Example:0.65')
            c=simpledialog.askfloat(title="Rotate Image",prompt="end row:")
        if(a==c):
            tkinter.messagebox.showinfo('Error','starting row and end row cannot be the same.')
            a=simpledialog.askfloat(title="Rotate Image",prompt="start row:")
            c=simpledialog.askfloat(title="Crop Image",prompt="end row:")
        if (c<a):
            tkinter.messagebox.showinfo('Error','endrow cannot smaller than startrow.')
            a=simpledialog.askfloat(title="Rotate Image",prompt="start row:")
            c=simpledialog.askfloat(title="Crop Image",prompt="end row:")
        d=simpledialog.askfloat(title="Crop Image",prompt="end column:")
        if (d>1 or d<=0):
            tkinter.messagebox.showinfo('Error','Please insert a integer in the range of 0<=x<=1. Example:0.65')
            d=simpledialog.askfloat(title="Rotate Image",prompt="end column:")
        if( d<b):
            tkinter.messagebox.showinfo('Error','endcolumn cannot smaller than startcolumn.')
            b=simpledialog.askfloat(title="Rotate Image",prompt="start column:")
            d=simpledialog.askfloat(title="Rotate Image",prompt="end column:")
               
        startRow=int(self.image.shape[0]*a)
        startCol=int(self.image.shape[1]*b)

        endRow=int(self.image.shape[0]*c)
        endCol=int(self.image.shape[1]*d)

        croppedImage=self.image[startRow:endRow,startCol:endCol]
        self.image=(croppedImage)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(croppedImage))
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

    def use_pen(self):
        self.activate_button(self.pen_button)
        self.size_multiplier = 1

    def use_brush(self):
        self.activate_button(self.brush_button)
        self.size_multiplier = 3

    def use_line(self):
        self.activate_button(self.line_button)

    def use_poly(self):
        self.activate_button(self.poly_button)
        self.size_multiplier = 2


    def choose_color(self):
        self.eraser_on = False
        color = askcolor(color=self.color)[1]
        if color is not None:
            self.color = color

    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)

    def activate_button(self, some_button, eraser_mode=False):
        self.set_status()
        if self.active_button:
            self.active_button.config(relief='raised')
        some_button.config(relief='sunken')
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.set_status(event.x, event.y)
        line_width = self.size_scale.get() * self.size_multiplier
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=line_width, fill=paint_color,
                               capstyle='round', smooth=True, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y


    def line(self, x, y):
        line_width = self.size_scale.get() * self.size_multiplier
        paint_color = 'white' if self.eraser_on else self.color
        self.canvas.create_line(self.line_start[0], self.line_start[1], x, y,
                           width=line_width, fill=paint_color,
                           capstyle='round', smooth=True, splinesteps=36)

    def point(self, event):
        self.set_status(event.x, event.y)
        btn = self.active_button["text"]
        if btn in ("Line", "Polygon"):
            self.size_multiplier = 1
            if any(self.line_start):
                self.line(event.x, event.y)
                self.line_start = ((None, None) if btn == 'Line'
                                   else (event.x, event.y))
            else:
                self.line_start = (event.x, event.y)

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def line_reset(self, event):
        self.line_start = (None, None)

    def color_default(self):
        self.color = self.DEFAULT_COLOR

    def set_status(self, x=None, y=None):
        if self.active_button:
            btn = self.active_button["text"]
            oldxy = (self.line_start if btn in ("Line", "Polygon")
                     else (self.old_x, self.old_y))

            self.var_status.set(f"Selected: {btn}\n" +
                                ((f"Old (x, y): {oldxy}\n(x, y): ({x}, {y})")
                                 if x is not None and y is not None else ""))
    
    

    def save_file(self):
        self.file=filedialog.asksaveasfilename(initialdir="mainproject/images",filetypes=(('PNG FILE','.png'),('png File')))
        self.file=self.file+".PNG"
        ImageGrab.grab().crop((0,120,1100,1000)).save(self.file)

    def save(self):
        self.image = Image.fromarray(self.image)
        self.image.save('image1.png')

    def close(self):
        if tkinter.messagebox.askyesno('Close program', 'Really quit'):
            self.root.quit()
        else:
            pass

if __name__ == '__main__':
    Paint()