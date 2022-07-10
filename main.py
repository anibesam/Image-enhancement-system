# IF WE ARE NOT FREE FROM SIN UNTIL WE DIE, JESUS IS NOT OUR SAVIOUR, then DEATH IS - Bill Johnson
#!/usr/bin/python
import sys
import os
import math
import numpy as np
import skimage.metrics
from skimage import measure
from tkinter import *
import tkinter as tk
# Matplot library
from matplotlib import pyplot as plt
from tkinter import filedialog, Button, LabelFrame, Label, Tk
# loading Python Imaging Library
from PIL import ImageTk, Image
# To get Menu when required
from tkinter import Menu
# From Messagebox
from tkinter import messagebox
# From CV2
from cv2 import cv2
# Create a window
root = Tk()

# Set Title as Image Loader
root.title("IMAGE ENHANCEMENT SYSTEM")

# Set the resolution of window
root.geometry("1080x900")
# Homomorphic filter class


class HomomorphicFilter:
    """Homomorphic filter implemented with diferents filters and an option to an external filter.

    High-frequency filters implemented:
        butterworth
        gaussian

    Attributes:
        a, b: Floats used on emphasis filter:
            H = a + b*H

        .
    """

    def __init__(self, a=0.5, b=1.5):
        self.a = float(a)
        self.b = float(b)

    # Filters
    def __butterworth_filter(self, I_shape, filter_params):
        P = I_shape[0]/2
        Q = I_shape[1]/2
        U, V = np.meshgrid(range(I_shape[0]), range(
            I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)**2+(V-Q)**2)).astype(float)
        H = 1/(1+(Duv/filter_params[0]**2)**filter_params[1])
        return (1 - H)

    def __gaussian_filter(self, I_shape, filter_params):
        P = I_shape[0]/2
        Q = I_shape[1]/2
        H = np.zeros(I_shape)
        U, V = np.meshgrid(range(I_shape[0]), range(
            I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)**2+(V-Q)**2)).astype(float)
        H = np.exp((-Duv/(2*(filter_params[0])**2)))
        return (1 - H)

    # Methods
    def __apply_filter(self, I, H):
        H = np.fft.fftshift(H)
        I_filtered = (self.a + self.b*H)*I
        return I_filtered

    def filter(self, I, filter_params, filter='butterworth', H=None):
        """
        Method to apply homormophic filter on an image

        Attributes:
            I: Single channel image
            filter_params: Parameters to be used on filters:
                butterworth:
                    filter_params[0]: Cutoff frequency 
                    filter_params[1]: Order of filter
                gaussian:
                    filter_params[0]: Cutoff frequency
            filter: Choose of the filter, options:
                butterworth
                gaussian
                external
            H: Used to pass external filter
        """

        #  Validating image
        if len(I.shape) is not 2:
            raise Exception('Improper image')

        # Take the image to log domain and then to frequency domain
        I_log = np.log1p(np.array(I, dtype="float"))
        I_fft = np.fft.fft2(I_log)

        # Filters
        if filter == 'butterworth':
            H = self.__butterworth_filter(
                I_shape=I_fft.shape, filter_params=filter_params)
        elif filter == 'gaussian':
            H = self.__gaussian_filter(
                I_shape=I_fft.shape, filter_params=filter_params)
        elif filter == 'external':
            print('external')
            if len(H.shape) is not 2:
                raise Exception('Invalid external filter')
        else:
            raise Exception('Selected filter not implemented')

        # Apply filter on frequency domain then take the image back to spatial domain
        I_fft_filt = self.__apply_filter(I=I_fft, H=H)
        I_filt = np.fft.ifft2(I_fft_filt)
        I = np.exp(np.real(I_filt))-1
        return np.uint8(I)
# End of class HomomorphicFilter

def homo_filt():
    # grab a reference to the image panels
    global panelA
    
    # open a file chooser dialog and allow the user to select an input
    # image
    path = filedialog.askopenfilename()
    # Running Path
    path_out = '/home/anibe/Desktop/augustine/'

    img_path_in = path
    img_path_out = path_out + 'filtered.png'

    # ensure a file path was selected
    if len(path) > 0:
        img = cv2.imread(img_path_in)[:, :, 0]
        homo_filter = HomomorphicFilter(a=0.75, b=1.25)
        img_filtered = homo_filter.filter(I=img, filter_params=[30, 2])
        cv2.imwrite(img_path_out, img_filtered)
    # convert the images to PIL format...
    edged = Image.fromarray(img_filtered)
    
     #GET ENTROPY
    def calcEntropy(img_filtered):
        entropy = []
        hist = cv2.calcHist([img_filtered], [0], None, [256], [0, 255])
        total_pixel = img_filtered.shape[0] * img.shape[1]
        for item in hist:
            probability = item / total_pixel
            if probability == 0:
                en = 0
            else:
                en = -1 * probability * (np.log(probability) / np.log(2))
                entropy.append(en)

                sum_en = np.sum(entropy)
                return sum_en


    if __name__ == '__main__':
        img1 = cv2.imread(img_path_out, cv2.IMREAD_GRAYSCALE)
        entropy1 = calcEntropy(img1)
        lbl = Label(root, fg="blue", text=(('Entropy:',entropy1)))
        lbl.pack(side="top", pady=6)
        print(entropy1)
    
    def psnr1(img_path_in, img_path_out):
        mse = np.mean((img/1.0 - img/1.0) ** 2 )
        if mse < 1.0e-10:
            return 100
        return 10 * math.log10(255.0**2/mse)
    if __name__ == '__main__':
        print(psnr1(img_path_in,img_path_out))
        print(skimage.metrics.peak_signal_noise_ratio(img, img_filtered, data_range=255))
        lbl = Label(root, fg="red", text=(('PNSR:',skimage.metrics.peak_signal_noise_ratio(img, img_filtered, data_range=255))))
        lbl.pack(side="top", pady=30)
    
    

    # ...and then to ImageTk format
    edged = ImageTk.PhotoImage(edged)
# if the panels are None, initialize them
    if panelA is None:
       
        # while the second panel will store the edge map
        panelA = Label(image=edged)
        panelA.image = edged
        panelA.pack(side="left", padx=10, pady=10)

        # otherwise, update the image panels
    else:
        # update the pannels
        panelA.configure(image=edged)
        panelA.image = edged


panelA = None

# TERMINATING HOMOMORPHIC FILTERING PART
# PART OF FUNCTIONS

# --- functions ---


def openFile():
    # open file dialog box to select image
    # The dialogue box has a title "Open"
    path = filedialog.askopenfilename()
    return path




def histoEqual():
    # grab a reference to the image panels
    global panelY
    # open a file chooser dialog and allow the user to select an input
    # image
    path = openFile()
    
    # Running Path
    path_out = '/home/anibe/Desktop/augustine/'

   
    heq_path_out = path_out + 'equalized.png'
    heq_path_in = path

    # ensure a file path was selected
    if len(path) > 0:
        img = cv2.imread(heq_path_in)[:, :, 0]
        # create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(img)
        cv2.imwrite(heq_path_out, img)

    # convert the images to PIL format...
    edged = Image.fromarray(cl1)
    
    #GET ENTROPY
    def calcEntropy2(img):
        entropy = []
        hist = cv2.calcHist([img], [0], None, [256], [0, 255])
        total_pixel = img.shape[0] * img.shape[1]
        for item in hist:
            probability = item / total_pixel
            if probability == 0:
                en = 0
            else:
                en = -1 * probability * (np.log(probability) / np.log(2))
                entropy.append(en)

                sum_en = np.sum(entropy)
                return sum_en

    if __name__ == '__main__':
        img1 = cv2.imread(heq_path_out, cv2.IMREAD_GRAYSCALE)
        entropy = calcEntropy2(img1)
        lbl = Label(root, fg="blue", text=(('Entropy:', entropy)))
        lbl.pack(side="top", pady=6)
        print(entropy)

    def psnr2(heq_path_out):
        mse = np.mean((img/255))
        if mse < 1.0e-10:
            return 100
        PIXEL_MAX = 1
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    if __name__ == '__main__':
        print(psnr2(heq_path_out))
        print(skimage.metrics.peak_signal_noise_ratio(img, img, data_range=255))
        lbl = Label(root, fg="red", text=(('PNSR:', psnr2(heq_path_out))))
        lbl.pack(side="top", pady=30)
   
    

    # ...and then to ImageTk format
    edged = ImageTk.PhotoImage(edged)

# if the panels are None, initialize them
    if panelY is None:
        # the first panel will store our original image
        panelY = Label(image=edged)
        panelY.image = edged
        panelY.pack(side="right", padx=10, pady=10)

        # otherwise, update the image panels
    else:
        # update the pannels
        panelY.configure(image=edged)
        panelY.image = edged

panelY = None

   



# TERMINATING HISTOGRAM EQUALIZATION
# PART OF FUNCTIONS

#ANIBE SAMUEL 
#------------------------------PROPOSED SYSTEM FUNCTION-----------------------------------------
def proposed_sys():
    # grab a reference to the image panels
    global panelG

    # open a file chooser dialog and allow the user to select an input
    # image
    path = filedialog.askopenfilename()
    # Running Path
    path_out = '/home/anibe/Desktop/augustine/'

    bt_path_in = path
    bt_path_out = path_out + 'combined.png'

    # ensure a file path was selected
    if len(path) > 0:
        img = cv2.imread(bt_path_in)[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # create a CLAHE object (Arguments are optional).
        cl1 = clahe.apply(img)
        homo_filter = HomomorphicFilter(a=0.75, b=1.25)
        img_filtered = homo_filter.filter(I=img, filter_params=[30, 2])
        both_calc = cl1 + img_filtered
        cv2.imwrite(bt_path_out, both_calc)
    # convert the images to PIL format...
    edged = Image.fromarray(both_calc)
    
     #GET COMBINATION ALGORITHM ENTROPY
    def calcEntropy(both_calc):
        entropy = []
        hist = cv2.calcHist([both_calc], [0], None, [256], [0, 255])
        total_pixel = both_calc.shape[0] * img.shape[1]
        for item in hist:
            probability = item / total_pixel
            if probability == 0:
                en = 0
            else:
                en = -1 * probability * (np.log(probability) / np.log(2))
                entropy.append(en)

                sum_en = np.sum(entropy)
                return sum_en


    if __name__ == '__main__':
        img1 = cv2.imread(bt_path_out, cv2.IMREAD_GRAYSCALE)
        entropy1 = calcEntropy(img1)
        lbl = Label(root, fg="blue", text=(('Entropy:',entropy1)))
        lbl.pack(side="top", pady=6)
        print(entropy1)
    
    def psnr1(bt_path_in, bt_path_out):
        mse = np.mean((img/1.0 - img/1.0) ** 2 )
        if mse < 1.0e-10:
            return 100
        return 10 * math.log10(255.0**2/mse)
    if __name__ == '__main__':
        print(psnr1(bt_path_in,bt_path_out))
        print(skimage.metrics.peak_signal_noise_ratio(img, both_calc, data_range=255))
        lbl = Label(root, fg="red", text=(('PNSR:',skimage.metrics.peak_signal_noise_ratio(img, both_calc, data_range=255))))
        lbl.pack(side="top", pady=30)
    
    
    

    # ...and then to ImageTk format
    edged = ImageTk.PhotoImage(edged)
# if the panels are None, initialize them
    if panelG is None:
        # the first panel will store our original image
        # while the second panel will store the edge map
        panelG = Label(image=edged)
        panelG.image = edged
        panelG.pack(side="right", padx=10, pady=10)

        # otherwise, update the image panels
    else:
        # update the pannels
        panelH.configure(image=edged)
        panelH.image = edged


panelG = None
panelH = None


    

#THIS TAKES YOU TO SSIM PROCESSS
def ssim():
    root.destroy()
    import ssim

def about():

    messagebox.showinfo('ABOUT IMAGE ENHANCEMENT SYSTEM',
                        'Image Enhancement System using General Histogram Equalization and Homomorphic Filtering to develop a hybrid system(combination of Both).')


def welcome():

    messagebox.showinfo('WELCOME',
                        'This App is developed by Anibe Sam http://twitter.com/anibesam on Twitter')

# Calling Documentation Button


def documentation():

    messagebox.showinfo(
        'DOCUMENTATION', 'We are currently working on our documentation')


# Calling Menu from Here
menu = Menu(root)
about_item = Menu(menu)
new_item = Menu(menu)
eva = Menu(menu)

# MENU BUTTON
new_item.add_command(label='Enhance Image with Histogram Equaliazion', command=histoEqual)
new_item.add_separator()
new_item.add_command(label='Enhance Image with Homomorphic Filtering', command=homo_filt)
new_item.add_separator()
new_item.add_command(label='Enhance Image Using Both Method', command=proposed_sys)
new_item.add_separator()
new_item.add_command(label='Exit Application', command=root.destroy)
menu.add_cascade(label='File', menu=new_item)

#EVALUATION MENU
eva.add_command(label='Get image SSIM and AMBE', command=ssim)
menu.add_cascade(label='Evaluation', menu=eva)

# HELP MENU
about_item.add_command(label='Welcome', command=welcome)
about_item.add_separator()
about_item.add_command(label='Documentation', command=documentation)
about_item.add_separator()
about_item.add_command(label='About', command=about)
menu.add_cascade(label='Help', menu=about_item)
root.config(menu=menu)

# ------------ BUTTONS -------------

# kick off the GUI
root.mainloop()
s
