from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QPixmap
from task import Ui_MainWindow
import sys
import random
import cv2
import numpy as np
import pyqtgraph as pg
from matplotlib.image import imread

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        path = 'lena.jpg'
        self.img = cv2.imread(path, 0)
        self.img2 = cv2.rotate(cv2.imread(path),cv2.ROTATE_90_CLOCKWISE)
        self.img3 = cv2.rotate(cv2.imread(path,0),cv2.ROTATE_90_CLOCKWISE)
        self.img4 = cv2.rotate(cv2.imread('test.jpg',0),cv2.ROTATE_90_CLOCKWISE)
        self.img5 = cv2.rotate(cv2.imread('test2.jpg',0),cv2.ROTATE_90_CLOCKWISE)
    
        ###########To remove axis###############
        self.ui.widget_2.getPlotItem().hideAxis('bottom')
        self.ui.widget_2.getPlotItem().hideAxis('left')
        self.ui.widget_3.getPlotItem().hideAxis('bottom')
        self.ui.widget_3.getPlotItem().hideAxis('left')
        self.ui.widget_4.getPlotItem().hideAxis('bottom')
        self.ui.widget_4.getPlotItem().hideAxis('left')

        self.Histogram()
        self.ui.image1.setPixmap(QPixmap(path))
        self.ui.comboBox_Image1_3.setEnabled(False)
        self.ui.comboBox_Image1.currentIndexChanged[int].connect(self.noisy_image)
        self.ui.comboBox_Image1_3.currentIndexChanged[int].connect(self.filtered_image)
        self.ui.comboBox_Image1_2.currentIndexChanged[int].connect(self.threshold_image)
        self.ui.comboBox.currentIndexChanged[int].connect(self.histogram_selection)
        self.ui.comboBox_4.currentIndexChanged[int].connect(self.visualize_hybrid_image)
        self.ui.comboBox_3.currentIndexChanged[int].connect(self.edge_detection_filter)
        self.ui.comboBox_2.currentIndexChanged[int].connect(self.Histogram)
        #self.get_histogram(self.img3,256)

    def convolution(self, image, kernel, average=False, verbose=False):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            None
        image_row, image_col = image.shape
        kernel_row, kernel_col = kernel.shape	 
        output = np.zeros(image.shape)	 
        pad_height = int((kernel_row - 1) / 2)
        pad_width = int((kernel_col - 1) / 2)	 
        padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))	 
        padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image	 
        for row in range(image_row):
            for col in range(image_col):
                output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
                if average:
                    output[row, col] /= kernel.shape[0] * kernel.shape[1]
        return output

    # Add additive noise to the image
    def uniform_noise(self, img):
        gaussian = np.random.randn(img.shape[0], img.shape[1])
        img = img + img*gaussian
        return img

    def gaussian_noise(self, img):
        mean = 0
        var = 100
        sigma = var ** 0.5
        gaussian = np.random.normal(mean, sigma, (256, 256))
        new_img = np.zeros(img.shape, np.float32)
        new_img = img + gaussian
        return new_img

    def salt_and_pepper_noise(self, img):
        row , col = img.shape
        # Randomly pick some pixels in the image for coloring them white
        number_of_pixels = random.randint(300, 10000)
        for i in range(number_of_pixels):	        
            # Pick a random y coordinate
            y_coord=random.randint(0, row - 1)	          
            # Pick a random x coordinate
            x_coord=random.randint(0, col - 1)	          
            # Color that pixel to white
            img[y_coord][x_coord] = 255	          
        # Randomly pick some pixels in the image for coloring them black
        number_of_pixels = random.randint(300 , 10000)
        for i in range(number_of_pixels):	        
            # Pick a random y coordinate
            y_coord=random.randint(0, row - 1)	          
            # Pick a random x coordinate
            x_coord=random.randint(0, col - 1)	          
            # Color that pixel to black
            img[y_coord][x_coord] = 0	          
        self.img = cv2.imread('lena.jpg', 0)
        return img
    
    # Filter the noisy image using the low pass filters

    def averaging_filter(self, img):
        row, col = img.shape
        # Develop Averaging filter(3, 3) mask
        mask = np.ones([3, 3], dtype = int)
        mask = mask / 9
        # Convolve the 3X3 mask over the image
        img_new = self.convolution(img, mask)
        return img_new

    def gaussian_filter(self, img):
        sigma = 0.5
        row, col = img.shape
        filter_size = 2 * int(4 * sigma + 0.5) + 1
        gaussian_filter = np.zeros((filter_size, filter_size), np.float32)
        m = filter_size//2
        n = filter_size//2
        for x in range(-m, m+1):
            for y in range(-n, n+1):
                x1 = 2*np.pi*(sigma**2)
                x2 = np.exp(-(x**2 + y**2)/(2* sigma**2))
                gaussian_filter[x+m, y+n] = x2/x1
        img_new = self.convolution(img, gaussian_filter)
        return img_new

    def median_filter(self, img):
        # Traverse the image. For every 3X3 area,find the median of the pixels and replace the center pixel by the median
        row, col = img.shape
        img_new = np.zeros([row, col])
        for i in range(1, row-1):
            for j in range(1, col-1):
                temp = [img[i-1, j-1], img[i-1, j], img[i-1, j + 1], img[i, j-1], img[i, j], img[i, j + 1], img[i + 1, j-1], img[i + 1, j], img[i + 1, j + 1]]
                temp = sorted(temp)
                img_new[i, j]= temp[4]
        return img_new

    def noisy_image(self):
        self.ui.comboBox_Image1_3.setEnabled(True)
        if self.ui.comboBox_Image1.currentIndex() == 1:
        	self.image = self.uniform_noise(self.img)
        elif self.ui.comboBox_Image1.currentIndex() == 2:
        	self.image = self.gaussian_noise(self.img)
        elif self.ui.comboBox_Image1.currentIndex() == 3:
        	self.image = self.salt_and_pepper_noise(self.img)
        self.ui.image2.setPixmap(QPixmap(self.display(self.image)))
        self.filtered_image()
        

    def filtered_image(self):
        if self.ui.comboBox_Image1_3.currentIndex() == 0:
            output = self.averaging_filter(self.image)
        elif self.ui.comboBox_Image1_3.currentIndex() == 1:
            output = self.gaussian_filter(self.image)
        elif self.ui.comboBox_Image1_3.currentIndex() == 2:
            output = self.median_filter(self.image)
        self.ui.image3.setPixmap(QPixmap(self.display(output)))

    def display(self, img):
            img = np.array(img).reshape(self.img.shape[1],self.img.shape[0]).astype(np.uint8)
            img = QtGui.QImage(img, img.shape[0] ,img.shape[1] ,QtGui.QImage.Format_Grayscale8)
            return img 

## High and Low pass Filters

    def high_pass_filter(self,img):
        row, col = img.shape
        mask = np.zeros((row, col))
        hpf = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
        mask[20:23,20:23] = hpf
        mask = np.fft.fftshift(np.fft.fft2(mask))
        image = np.fft.fftshift(np.fft.fft2(img))
        img_new = np.fft.ifft2(mask*image)
        img_new = np.log(1+np.abs(img_new))
        return img_new

    def low_pass_filter(self, img):
        row, col = img.shape
        mask = np.zeros((row, col))
        avgMask= np.ones((9,9))/81
        mask[24:33,24:33] = avgMask
        mask = np.fft.fftshift(np.fft.fft2(mask))
        image = np.fft.fftshift(np.fft.fft2(img))
        img_new = np.fft.ifft2(mask*image)
        img_new = np.log(1+np.abs(img_new))
        return img_new

## Edge Detection

# Sobel Edge detection
    def sobel(self,img):
        row, col = img.shape
        Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
        Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        self.new_img = np.zeros([row, col]) 
        for i in range(row - 2):
            for j in range(col - 2):
                gx = np.sum(np.multiply(Gx, img[i:i + 3, j:j + 3]))  # x direction
                gy = np.sum(np.multiply(Gy, img[i:i + 3, j:j + 3]))  # y direction
                self.new_img[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)
        self.img_new = self.new_img.astype(np.uint8)    
        return self.img_new

#Prewitt Edge Detection
    def perwitt(self,img) :
        row, col = img.shape
        Gx = np.array([[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]])
        Gy = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]])
        self.new_img = np.zeros([row, col]) 
        for i in range(row - 2):
            for j in range(col - 2):
                gx = np.sum(np.multiply(Gx, img[i:i + 3, j:j + 3]))  # x direction
                gy = np.sum(np.multiply(Gy, img[i:i + 3, j:j + 3]))  # y direction
                self.new_img[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)
        self.img_new = self.new_img.astype(np.uint8)    
        return self.img_new

#Roberts Edge Detection
    def roberts(self,img):
        row, col = img.shape
        Gx = np.array([[1.0, 0.0], [0.0, -1.0]])
        Gy = np.array([[0.0, -1.0], [1.0, 0.0]])
        self.new_img = np.zeros([row, col]) 
        for i in range(row - 2):
            for j in range(col - 2):
                gx = np.sum(np.multiply(Gx, img[i:i + 2, j:j + 2]))  # x direction
                gy = np.sum(np.multiply(Gy, img[i:i + 2, j:j + 2]))  # y direction
                self.new_img[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)
        self.img_new = self.new_img.astype(np.uint8)    
        return self.img_new
 
    def edge_detection_filter(self):
        if self.ui.comboBox_3.currentIndex() == 1:
            output = self.sobel(self.img3)
        elif self.ui.comboBox_3.currentIndex() == 2:
            output = self.perwitt(self.img3)
        elif self.ui.comboBox_3.currentIndex() == 3:
            output = self.roberts(self.img3)
        elif self.ui.comboBox_3.currentIndex() == 4:
            output = self.high_pass_filter(self.img3)
        elif self.ui.comboBox_3.currentIndex() == 5:
            output = self.low_pass_filter(self.img3)
        #self.ui.widget_4.setPixmap(QPixmap(self.display(output)))
        self.my_img = pg.ImageItem(output)
        self.ui.widget_4.addItem(self.my_img)

########################Threshold################################
    def global_threshold_v_127(self,img):
        thresh = 100
        binary = img > thresh
        for i in range(0,len(binary),1):
            for j in range(0,len(binary),1):
                if binary[i][j] == True:
                    binary[i][j] = 256
                else:
                    binary[i][j]=0

        return binary

    def local_treshold(self,input_img):
        h, w = input_img.shape
        S = w/8
        s2 = S/2
        T = 15.0
        #integral img
        int_img = np.zeros_like(input_img, dtype=np.uint32)
        for col in range(w):
            for row in range(h):
                int_img[row,col] = input_img[0:row,0:col].sum()
        #output img
        out_img = np.zeros_like(input_img)    
        for col in range(w):
            for row in range(h):
                #SxS region
                y0 = int(max(row-s2, 0))
                y1 = int(min(row+s2, h-1))
                x0 = int(max(col-s2, 0))
                x1 = int(min(col+s2, w-1))
                count = (y1-y0)*(x1-x0)
                sum_ = int_img[y1, x1]-int_img[y0, x1]-int_img[y1, x0]+int_img[y0, x0]
                if input_img[row, col]*count < sum_*(100.-T)/100.:
                    out_img[row,col] = 0
                else:
                    out_img[row,col] = 255
        return out_img

    

    def threshold_image(self):
        if self.ui.comboBox_Image1_2.currentIndex() == 1:
            out_put = (self.global_threshold_v_127(self.img3))
            my_img = pg.ImageItem(out_put)
            self.ui.widget_2.addItem(my_img)
        elif self.ui.comboBox_Image1_2.currentIndex() == 2:
            out = self.local_treshold(self.img3)
            my_img = pg.ImageItem(out)
            self.ui.widget_2.addItem(my_img)
        else:
            pass


    
    #######################Histograms#################################
    def Transormation_to_grayScale(self,input_image):
        H,W = input_image.shape[:2]
        gray = np.zeros((H,W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i,j] = np.clip(0.07 * input_image[i,j,0]  + 0.72 * input_image[i,j,1] + 0.21 * input_image[i,j,2], 0, 255)
        return gray

    def cumulative_histogram(self,hist_input):
        hist= iter(hist_input)
        b = [next(hist)]
        for i in hist:
            b.append(b[-1] + i)
        
        b = np.array(b)
        nj = (b - b.min()) * 255
        N = b.max() - b.min()

        cum_curv = nj/N
        return cum_curv


    def get_histogram(self, image_input,bins):
        # array with size of bins, set to zeros
        histogram = np.zeros(bins)
        
        # loop through pixels and sum up counts of pixels
        for pixel in image_input:
            histogram[pixel] += 1
        
        y=np.linspace(0,np.max(histogram))
        # return our final result
        return histogram
        

    def histogram_selection(self):
        if self.ui.comboBox.currentIndex() == 5:
            self.ui.widget.clear()
            cum_curve = self.cumulative_histogram(self.get_histogram(self.img3,256))
            #curve = pg.plot(cum_curve)
            self.ui.widget.plot(cum_curve)
        elif self.ui.comboBox.currentIndex() == 2:
            self.ui.widget.clear()
            red_hist = self.get_histogram(self.img2[:,:,0],256)
            my_img = pg.BarGraphItem(x=red_hist,height=self.img2[:,:,0].flatten(),width=0.6, brush='r')
            self.ui.widget.addItem(my_img)
        elif self.ui.comboBox.currentIndex() == 3:
            self.ui.widget.clear()
            green_hist = self.get_histogram(self.img2[:,:,1],256)
            my_img = pg.BarGraphItem(x=green_hist,height=self.img2[:,:,1].flatten(),width=0.6, brush='g')
            self.ui.widget.addItem(my_img)
            #pg.HistogramLUTWidgetm()
        elif self.ui.comboBox.currentIndex() == 4:
            self.ui.widget.clear()
            blue_hist = self.get_histogram(self.img2[:,:,2],256)
            my_img = pg.BarGraphItem(x=blue_hist,height=self.img2[:,:,2].flatten(),width=0.6, brush='g')
            self.ui.widget.addItem(my_img)
        elif self.ui.comboBox.currentIndex() == 1:
            self.ui.widget.clear()
            gray_image = self.Transormation_to_grayScale(self.img2)
            my_img = pg.ImageItem(gray_image)
            self.ui.widget.addItem(my_img)
        elif self.ui.comboBox.currentIndex() ==6:
            self.ui.widget.clear()
            cum_hist = self.cumulative_histogram(self.get_histogram(self.img3,256))
            cum_hist = cum_hist.astype('uint8')
            histo = cum_hist[self.img3.flatten()]
            my_img = pg.BarGraphItem(x=histo,height=self.img2[:,:,0].flatten(),width=0.6, brush='r') 
            self.ui.widget.addItem(my_img)



        else:
            pass



    ############################Hybrid Image###########################  
    def laplacian_image(self,input_image):
        input_image -= np.amin(input_image) #map values to the (0, 255) range
        A =input_image* 255.0/np.amax(input_image)

        #Kernel for negative Laplacian
        kernel = np.ones((3,3))*(-1)
        kernel[1,1] = 8

        #Convolution of the image with the kernel:
        Lap = self.convolution(A, kernel)

        return Lap


    def Hybrid_image(self,first_image,second_image,alpha):
        if first_image.shape[0] == second_image.shape[0] and first_image.shape[1] == second_image.shape[1]:
            laplace_image = self.laplacian_image(first_image)
            gaussian_image = self.gaussian_filter(second_image)
            Hybrid_img = alpha * laplace_image + (1 - alpha) * gaussian_image
            return Hybrid_img
    
    def Hybrid_image_frequency(self,first_image,second_image,alpha):
        if first_image.shape[0] == second_image.shape[0] and first_image.shape[1] == second_image.shape[1]:
            high_image = self.high_pass_filter(first_image)
            low_image = self.low_pass_filter(second_image)
            Hybrid_image = alpha * high_image + (1-alpha) * low_image
        return Hybrid_image

    def visualize_hybrid_image(self):
        if self.ui.comboBox_4.currentIndex() == 3:
            #self.ui.widget_3.clear()
            Hybrid_img = self.Hybrid_image(self.img4,self.img5,0.3)
            my_img = pg.ImageItem(Hybrid_img)
            self.ui.widget_3.addItem(my_img)

        elif self.ui.comboBox_4.currentIndex() == 4:
            Hybrid_img = self.Hybrid_image_frequency(self.img4,self.img5,0.3)
            my_img = pg.ImageItem(Hybrid_img)
            self.ui.widget_3.addItem(my_img)

        elif self.ui.comboBox_4.currentIndex() == 2:
            self.ui.widget_3.clear()
            my_img = pg.ImageItem(self.img4)
            self.ui.widget_3.addItem(my_img)
        elif self.ui.comboBox_4.currentIndex() == 1:
            self.ui.widget_3.clear()
            my_img = pg.ImageItem(self.img5)
            self.ui.widget_3.addItem(my_img)

#######################Histogram###########################    
    def Histogram(self):
    	self.ui.widget_5.clear()
    	if self.ui.comboBox_2.currentIndex() == 0:
    		hist =self.Hist(self.img)
    		self.ui.widget_5.plot(hist.flatten())
    	elif self.ui.comboBox_2.currentIndex() == 1:
    		image = self.eqimage(self.img3)
    		image = pg.ImageItem(image)
    		self.ui.widget_5.addItem(image)
    	elif self.ui.comboBox_2.currentIndex() == 2:
    		norm = self.normalization(self.img3)
    		norm = pg.ImageItem(norm)
    		self.ui.widget_5.addItem(norm)
    
    def Hist(self,image):
    	H = np.zeros(shape = (256,1))
    	s = image.shape
    	for i in range (s[0]):
    		for j in range(s[1]):
    			k=image[i,j]
    			H[k,0]+=1
    	return H

    def eqimage(self,image):
    	s = image.shape
    	H = self.Hist(image)
    	x = H.reshape(1,256)
    	y = np.array([])
    	y=np.append(y,x[0,0])
    	for i in range (255):
    		k = x[0,i+1]+y[i]
    		y = np.append(y,k)
    	y = np.round((y/(s[0]*s[1]))*255)
    	for i in range(s[0]):
    		for j in range (s[1]):
    			k = image[i,j]
    			image[i,j] = y[k]
    	self.img3 = cv2.rotate(cv2.imread("lena.jpg",0),cv2.ROTATE_90_CLOCKWISE)
    	return image

    def eqHist(self,image):
	    img = self.eqimage(image)
	    eqhsit = self.Hist(img)
	    return eqhsit

    def normalization(self,Image):
    	pixels = Image.astype('float32')
    	pixels = pixels * (40/255) +10
    	return pixels


def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()
