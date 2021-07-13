from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QPixmap,QImage
from PyQt5.QtCore import Qt
from task import Ui_MainWindow
import sys
import random
import cv2
import snake as sn
import canny
import math
import hough
import pyqtgraph as pg
import numpy as np


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.clear = False
        self.ui.image_1.getPlotItem().hideAxis('bottom')
        self.ui.image_1.getPlotItem().hideAxis('left')

        file_to_load = "images/example.jpg"
        file_to_load2 = "images/example2.jpg"
        self.image_2 = cv2.rotate(cv2.imread( file_to_load2, cv2.IMREAD_COLOR ),cv2.ROTATE_90_CLOCKWISE)
        path = 'circle.jpg'
        self.img = cv2.imread(path, 0)
        #self.img3 = cv2.rotate(cv2.imread(path,0),cv2.ROTATE_90_CLOCKWISE)
        self.image_1 = cv2.rotate(cv2.imread( file_to_load, cv2.IMREAD_COLOR ),cv2.ROTATE_90_CLOCKWISE)
        # self.ui.pushButton.clicked.connect(self.open_canny_tab)
        # self.ui.pushButton_2.clicked.connect(self.open_Hough_tab)
        # self.ui.pushButton_3.clicked.connect(self.open_Active_contour_tab)
        self.ui.image1.setPixmap(QPixmap(path))
        self.ui.comboBox_3.currentIndexChanged[int].connect(self.canny_edge_detector)
        self.ui.pushButton_4.clicked.connect(self.active_contour)
        self.ui.pushButton_5.clicked.connect(self.clearbutton)
        self.ui.comboBox.currentIndexChanged[int].connect(self.houghlines)

    def open_canny_tab(self):
        self.ui.tabWidget.setCurrentIndex(0)

    def open_Hough_tab(self):
        self.ui.tabWidget.setCurrentIndex(1)

    def open_Active_contour_tab(self):
        self.ui.tabWidget.setCurrentIndex(2)

    def clearbutton(self):
        self.clear = True
        self.ui.image_1.clear()

    def startbutton(self):
        self.clearbutton = False

    def active_contour(self):
        self.clear = False
        snake = sn.Snake( self.image_1, closed = True )
        
        while(True):
            snakeImg = snake.visuaize_Image()
            img = pg.ImageItem(snakeImg)
            self.ui.image_1.addItem(img)
            x = []
            y = []
            for i in range(len(snake.points)):
            	x.append(snake.points[i][0])
            	y.append(snake.points[i][1])
            area=0.5*np.sum(y[:-1]*np.diff(x) - x[:-1]*np.diff(y))
            area=np.abs(area)
            perimeter = snake.get_length()
            self.ui.textEdit.setPlaceholderText("{}".format(area/10000))
            self.ui.textEdit_2.setPlaceholderText("{}".format(perimeter/100))
            # self.ui.image_1.setTitle("area = {} , perimeter = {}".format(area, perimeter))
            snake_changed = snake.step()
            
            self.ui.slider_1.valueChanged[int].connect(snake.set_alpha)
            self.ui.slider_2.valueChanged[int].connect(snake.set_beta)
            self.ui.slider_3.valueChanged[int].connect(snake.set_gamma)

            k = cv2.waitKey(33)
            if self.clear == True:
                if k == 27:
                     break
                cv2.destroyAllWindows()
                break


    def houghlines(self):
        if self.ui.comboBox.currentIndex() == 1:
            original = pg.ImageItem(self.image_2)
            self.ui.widget_10.addItem(original)
        elif self.ui.comboBox.currentIndex() == 2:
            acc,_,_,_ = hough.accumulator(self.image_2)
            x = pg.ImageItem(acc)
            self.ui.widget_3.addItem(x)
        elif self.ui.comboBox.currentIndex() == 3:
            hough.lines(self.image_2)
            y = cv2.imread('lines.png')
            z = pg.ImageItem(y)
            self.ui.widget_5.addItem(z )

    def canny_edge_detector(self):

        if self.ui.comboBox_3.currentIndex() == 1:
            sigma = 0.5
            T = 0.3
            x, y = canny.MaskGeneration(T, sigma)
            gauss =canny.Gaussian(x, y, sigma)
            gx = canny.Create_Gx(x, y)
            gy = canny.Create_Gy(x, y)
            image = cv2.imread('circle.jpg')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            smooth_img = canny.smooth(gray, gauss)
            fx = canny.ApplyMask(smooth_img, gx)
            fy = canny.ApplyMask(smooth_img, gy)
            mag = canny.Gradient_Magnitude(fx, fy)
            mag = mag.astype(int)
            Angle = canny.Gradient_Direction(fx, fy)
            quantized = canny.Digitize_angle(Angle)
            nms = canny.Non_Max_Supp(quantized, Angle, mag)
            threshold = canny._double_thresholding(nms, 30, 60)
            hys = canny._hysteresis(threshold)
        self.my_img = pg.ImageItem(hys)
        self.ui.image.addItem(self.my_img)




def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()




if __name__ == "__main__":
    main()