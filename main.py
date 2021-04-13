from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QPixmap
from task import Ui_MainWindow
import sys
import random


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.pushButton.clicked.connect(self.open_canny_tab)
        self.ui.pushButton_2.clicked.connect(self.open_Hough_tab)
        self.ui.pushButton_3.clicked.connect(self.open_Active_contour_tab)

    def open_canny_tab(self):
        self.ui.tabWidget.setCurrentIndex(0)

    def open_Hough_tab(self):
        self.ui.tabWidget.setCurrentIndex(1)

    def open_Active_contour_tab(self):
        self.ui.tabWidget.setCurrentIndex(2)




def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()