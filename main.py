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
        self.image_1 = cv2.imread( file_to_load, cv2.IMREAD_COLOR )
        # self.ui.pushButton.clicked.connect(self.open_canny_tab)
        # self.ui.pushButton_2.clicked.connect(self.open_Hough_tab)
        # self.ui.pushButton_3.clicked.connect(self.open_Active_contour_tab)
        self.ui.pushButton_4.clicked.connect(self.active_contour)
        self.ui.pushButton_5.clicked.connect(self.clearbutton)

    def open_canny_tab(self):
        self.ui.tabWidget.setCurrentIndex(0)

    def open_Hough_tab(self):
        self.ui.tabWidget.setCurrentIndex(1)

    def open_Active_contour_tab(self):
        self.ui.tabWidget.setCurrentIndex(2)

    def clearbutton(self):
        self.clear = True

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

            # Stops looping when ESC pressed
            k = cv2.waitKey(33)
            if self.clear == True:
                if k == 27:
                     break
                cv2.destroyAllWindows()
                break


def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()