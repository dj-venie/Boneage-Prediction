# -*- coding: utf-8 -*-
from PyQt5 import QtGui, QtCore, QtWidgets, uic
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cv2
import sys
import numpy as np
import img_processing
from tensorflow.keras.models import load_model
import patient

MainUI = "./main.ui"

app = QApplication(sys.argv)
file = QtCore.QFile("./qss/MaterialDark.qss")
file.open(QtCore.QFile.ReadOnly | QtCore.QFile.Text)
stream = QtCore.QTextStream(file)
app.setStyleSheet(stream.readAll())

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(MainUI, self)
        # self.setWindowFlag(QtCore.Qt.FramelessWindowHint)

        ###Tab 1(Main)
        self.image = None #원본 이미지
        self.roi = None #ROI 이미지
        self.output = None #골연령 예측 결과
        self.gender = np.array([1]) #성별. 화면에서 Male 체크를 기본으로 함. 입력변수 Female [0], Male [1]
        self.model = load_model('./model/sj_vgg_variation_new_real.h5')
        self.model.load_weights('./model/sj_vgg_variation_new_real_weight.h5')

        #초기 화면에서 sample 이미지 보여주기
        self.qPixmapFile_label_origin = QPixmap()
        self.qPixmapFile_label_origin.load("./image/sample_origin.png")
        self.qPixmapFile_label_origin = self.qPixmapFile_label_origin.scaled(450, 500)
        self.label_origin.setPixmap(self.qPixmapFile_label_origin)

        self.qPixmapFile_label_filter = QPixmap()
        self.qPixmapFile_label_filter.load("./image/sample_filter1.png")
        self.qPixmapFile_label_filter = self.qPixmapFile_label_filter.scaled(450, 500)
        self.label_filter.setPixmap(self.qPixmapFile_label_filter)

        #버튼 클릭시 함수 연결
        self.pushButton_upload.clicked.connect(self.openFileNameDialog) #Upload
        self.pushButton_filter1.clicked.connect(self.filter_contrast) #Filter 1
        self.pushButton_filter2.clicked.connect(self.filter_equalization) #Filter 2
        self.pushButton_filter3.clicked.connect(self.filter_sobel) #Filter 3
        self.pushButton_customize.clicked.connect(self.customize_clicked) #Customize
        self.radioButton_male.clicked.connect(self.gender_checked) #Male
        self.radioButton_female.clicked.connect(self.gender_checked) #Female
        self.pushButton_extract.setEnabled(False) #초기 화면에서 Exract ROI 버튼은 비활성화하고, 이미지 업로드 후 활성화
        self.pushButton_extract.clicked.connect(self.extract_roi_clicked)


        ###Tab 2(List)
        self.header_list = ["Patient ID", "Examination No.", "Name", "Sex", "Birth Date", "Examination Date", "Image", "Bone-Age", "Load to Main"] #표의 열이름
        self.tableWidget.setHorizontalHeaderLabels(self.header_list)
        self.lineEdit_id.textChanged.connect(self.lineeditTextFunction) #Patient ID 검색창
        self.lineEdit_id.returnPressed.connect(self.show_list_by_id) #Patient ID 검색시 함수 연결
        self.pushButton_search.clicked.connect(self.show_list_by_id) #Search
        self.pushButton_all.clicked.connect(self.show_all_list) #Search All

    def openFileNameDialog(self):  # opencv로 이미지 불러와서 qimage resize로 사이즈 줄인 후 label에 보여주기.
        fileName, _ = QFileDialog.getOpenFileName(self, "파일 선택", "",
                                                  "Image files (*.jpg *.gif *.png)")
        if fileName:
            self.setWindowTitle(fileName)
            self.progressBar_main.setValue(0)
            self.fileName = str(fileName)
            self.image = cv2.imdecode(np.fromfile(self.fileName, np.uint8), cv2.IMREAD_COLOR) #cv2.imread()
            self.label_origin_show(self.image)
            self.mask = img_processing.mask(self.image)
            self.image_filter1 = img_processing.contrast(self.image, self.mask)
            self.image_filter2 = img_processing.equalization(self.image, self.mask)
            self.image_filter3 = img_processing.sobel(self.image, self.mask)
            self.roi = img_processing.roi_carpal_bone(self.image, self.mask)
            self.label_filter_show(self.image_filter1)
            self.label_roi_show(self.roi)
            self.output = self.bone_age_pred(self.gender, self.roi)
            self.label_prediction_show(self.output)
            self.pushButton_extract.setEnabled(True)
            self.progressBar_main.setValue(100)

    #좌측 화면(label_origin)에 이미지 보여주는 함수
    def label_origin_show(self, image):
        self.img = QtGui.QImage(image.data, image.shape[1], image.shape[0], image.shape[1]*3,
                                    QtGui.QImage.Format_RGB888).rgbSwapped()
        self.img = QtGui.QImage(self.img).scaled(450, 500, QtCore.Qt.KeepAspectRatio)  # GUI에 보여주기 위한 용도로 사진 줄이기
        self.label_origin.setPixmap(QtGui.QPixmap.fromImage(self.img))

    #우측 화면(label_filter)에 이미지 보여주는 함수
    def label_filter_show(self, image):
        self.img = QtGui.QImage(image.data, image.shape[1], image.shape[0], image.shape[1]*3,
                                    QtGui.QImage.Format_RGB888).rgbSwapped()
        self.img = QtGui.QImage(self.img).scaled(450, 500, QtCore.Qt.KeepAspectRatio)
        self.label_filter.setPixmap(QtGui.QPixmap.fromImage(self.img))

    #아래 좌측 화면(label_roi)에 이미지 보여주는 함수
    def label_roi_show(self, image):
        self.img = QtGui.QImage(image.data, image.shape[1], image.shape[0], image.shape[1]*3,
                                QtGui.QImage.Format_RGB888).rgbSwapped()
        self.img = QtGui.QImage(self.img).scaled(150, 200, QtCore.Qt.KeepAspectRatio)
        self.label_roi.setPixmap(QtGui.QPixmap.fromImage(self.img))

    #contrast 이미지 보여주기
    def filter_contrast(self):
        if self.image is not None:
            self.label_filter_show(self.image_filter1)
        else:
            self.qPixmapFile_label_filter = QPixmap()
            self.qPixmapFile_label_filter.load("./image/sample_filter1.png")
            self.qPixmapFile_label_filter = self.qPixmapFile_label_filter.scaled(450, 500)
            self.label_filter.setPixmap(self.qPixmapFile_label_filter)

    #equalization 이미지 보여주기
    def filter_equalization(self):
        if self.image is not None:
            self.label_filter_show(self.image_filter2)
        else:
            self.qPixmapFile_label_filter = QPixmap()
            self.qPixmapFile_label_filter.load("./image/sample_filter2.png")
            self.qPixmapFile_label_filter = self.qPixmapFile_label_filter.scaled(450, 500)
            self.label_filter.setPixmap(self.qPixmapFile_label_filter)

    #sobel 이미지 보여주기
    def filter_sobel(self):
        if self.image is not None:
            self.label_filter_show(self.image_filter3)
        else:
            self.qPixmapFile_label_filter = QPixmap()
            self.qPixmapFile_label_filter.load("./image/sample_filter3.png")
            self.qPixmapFile_label_filter = self.qPixmapFile_label_filter.scaled(450, 500)
            self.label_filter.setPixmap(self.qPixmapFile_label_filter)

    #Male, Female 체크시 self.gender 값 변경
    def gender_checked(self):
        if self.radioButton_male.isChecked(): #Male 체크인 경우, [1]
            self.gender = np.array([1])
        else: #Female 체크인 경우, [2]
            self.gender = np.array([0])

        if self.roi is not None: #이미지 업로드 후 성별 변경시 골연령 다시 예측
            self.progressBar_main.setValue(0)
            self.output = self.bone_age_pred(self.gender, self.roi)
            self.label_prediction_show(self.output)
            self.progressBar_main.setValue(100)

    #골연령 예측 모형
    def bone_age_pred(self, gender, roi): #gender, roi 모두 array 타입
        roi = cv2.resize(roi, (251, 251), interpolation=cv2.INTER_AREA).reshape(-1, 251, 251, 3)
        input = [gender, roi]
        prediction = self.model.predict(input)
        pred_month = prediction * 17.941051615527474 + 61.39948849104856
        pred_year = pred_month / 12
        output = str(round(pred_year[0][0], 1))
        return output

    #골연령 예측 결과 보여주기(Main)
    def label_prediction_show(self, output):
        self.label_prediction.setText("Bone-Age : " + output + " years")

    #customize로 canny edge 추출
    def customize_clicked(self):
        cv2.namedWindow('Customize', cv2.WINDOW_NORMAL)

        cv2.createTrackbar('Low', 'Customize', 0, 1000, self.nothing)
        cv2.createTrackbar('High', 'Customize', 0, 1000, self.nothing)

        # 트랙바 초기값 지정
        cv2.setTrackbarPos('Low', 'Customize', 50)
        cv2.setTrackbarPos('High', 'Customize', 150)

        if self.image is not None:
            self.img_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            self.img_gray = cv2.imread("./image/sample_origin.png", cv2.IMREAD_GRAYSCALE)
        self.img_gray = cv2.GaussianBlur(self.img_gray, (3, 3), 0)
        while True:
            self.low = cv2.getTrackbarPos('Low', 'Customize')
            self.high = cv2.getTrackbarPos('High', 'Customize')
            self.img_canny = cv2.Canny(self.img_gray, self.low, self.high)
            cv2.imshow('Customize', self.img_canny)
            if cv2.waitKey(1) & 0xFF == 27: #Esc로 종료
                break
        cv2.destroyAllWindows()

    def nothing(self, x):
        pass

    #extract ROI
    def extract_roi_clicked(self):
        if self.image is not None:
            self.isDragging = False
            self.x0, self.y0, self.w, self.h = -1, -1, -1, -1
            cv2.namedWindow("Extract ROI", cv2.WINDOW_NORMAL)
            cv2.imshow("Extract ROI", self.image)
            cv2.setMouseCallback("Extract ROI", self.onMouse)
            cv2.waitKey()
            cv2.destroyAllWindows()

    def onMouse(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN: #왼쪽 마우스 버튼 다운, 드래그 시작
            self.isDragging = True
            self.x0 = x
            self.y0 = y
        elif event == cv2.EVENT_MOUSEMOVE: #마우스 움직임
            if self.isDragging:
                img_draw = self.image.copy()
                cv2.rectangle(img_draw, (self.x0, self.y0), (x, y), (0,255,0), 3)
                cv2.imshow("Extract ROI", img_draw)
        elif event == cv2.EVENT_LBUTTONUP: #왼쪽 마우스 버튼 업, 드래그 중지
            if self.isDragging:
                self.isDragging = False
                self.w = x - self.x0
                self.h = y - self.y0
                if self.w > 0 and self.h > 0:
                    img_draw = self.image.copy()
                    cv2.rectangle(img_draw, (self.x0, self.y0), (x, y), (0,255,0), 3)
                    cv2.imshow("Extract ROI", img_draw)
                    roi_extracted = self.image[self.y0:self.y0+self.h, self.x0:self.x0+self.w]
                    self.progressBar_main.setValue(0)
                    self.roi = img_processing.filter_roi_carpal(roi_extracted)
                    self.label_roi_show(self.roi)
                    self.output = self.bone_age_pred(self.gender, self.roi)
                    self.label_prediction_show(self.output)
                    self.progressBar_main.setValue(100)
                else:
                    cv2.imshow("Extract ROI", self.image)


    ####################################################################################################
    ###Tab 2(List)

    #Patient ID 검색창에 입력받은 텍스트 보여주기
    def lineeditTextFunction(self):
        self.lineEdit_id.setText(self.lineEdit_id.text())

    #테이블에 환자 리스트 보여주기
    def show_list(self, patient_list):

        self.tableWidget.clearContents()
        count = len(patient_list)
        self.tableWidget.setRowCount(count)
        self.gender_list = [0]*count #환자별 성별을 저장할 리스트. gender_list의 인덱스는 tableWidget의 행 인덱스와 동일.
        self.img_list = [0]*count #환자별 Image를 저장할 리스트. img_list의 인덱스는 tableWidget의 행 인덱스와 동일.

        #DB에서 가져온 데이터를 tableWidget에 보여주기
        nrow = 0
        for tup in patient_list:
            ncol = 0
            for data in tup:
                if ncol == 3: # 열 인덱스가 3인 경우(Column: Sex)
                    if str(data) == "Male":
                        self.gender_list[nrow] = np.array([1])
                    else:
                        self.gender_list[nrow] = np.array([0])
                    data_ = QTableWidgetItem(str(data))
                    self.tableWidget.setItem(nrow, ncol, data_)
                elif ncol == 6: #열 인덱스가 6인 경우(Column: Image)
                    self.btn_view = QtWidgets.QPushButton("View")  #View 버튼 생성
                    self.btn_view.clicked.connect(self.btn_view_clicked)
                    self.tableWidget.setCellWidget(nrow, ncol, self.btn_view)
                    img_data = np.fromstring(data.read(), dtype=np.uint8) #data는 oracle LOB 객체. fromstring을 통해 ndarray로 변환.
                    img_data = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                    self.img_list[nrow] = img_data #tableWidget에 string만 저장 가능해서, 이미지는 img_list 리스트에 별도로 저장
                elif ncol == 7: #열 인덱스가 7인 경우(Column: Bone-Age)
                    if str(data) == "None":
                        gender_ = self.gender_list[nrow]
                        img_ = self.img_list[nrow]
                        mask_ = img_processing.mask(img_)
                        roi_ =  img_processing.roi_carpal_bone(img_, mask_)
                        output_ = self.bone_age_pred(gender_, roi_)
                        data_ = QTableWidgetItem(str(output_))
                        self.tableWidget.setItem(nrow, ncol, data_)
                        #예측된 골연령을 DB에 update
                        patient_id_ = self.tableWidget.item(nrow, 0).text()
                        examination_number_ = self.tableWidget.item(nrow, 1).text()
                        patient.update_bone_age(str(output_), patient_id_, examination_number_)
                    else:
                        data_ = QTableWidgetItem(str(data))
                        self.tableWidget.setItem(nrow, ncol, data_)
                else:
                    data_ = QTableWidgetItem(str(data))
                    self.tableWidget.setItem(nrow, ncol, data_)
                ncol += 1

            #Load to Main 버튼 만들기
            self.btn_loadtomain = QtWidgets.QPushButton("Load")
            self.btn_loadtomain.clicked.connect(self.btn_loadtomain_clicked)
            self.tableWidget.setCellWidget(nrow, 8, self.btn_loadtomain)
            nrow += 1

    #Patient ID 검색시 리스트 보여주기
    def show_list_by_id(self):
        self.progressBar_list.setValue(0)
        self.patient_id = self.lineEdit_id.text()
        patient_list = patient.get_list_by_id(self.patient_id)
        self.show_list(patient_list)
        self.progressBar_list.setValue(100)

    #전체 리스트 보여주기
    def show_all_list(self):
        self.progressBar_list.setValue(0)
        patient_list = patient.get_all_list()
        self.show_list(patient_list)
        self.progressBar_list.setValue(100)

    #선택한 행(환자)의 이미지 보여주기
    def btn_view_clicked(self):
        nrow = self.tableWidget.currentRow() #현재 선택하고 있는 항목의 행을 반환
        img = self.img_list[nrow]
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    #선택한 행(환자)의 이미지를 Main 페이지에 보여주기
    def btn_loadtomain_clicked(self):
        nrow = self.tableWidget.currentRow() #현재 선택하고 있는 항목의 행을 반환
        self.setWindowTitle("Patient ID: " + self.tableWidget.item(nrow, 0).text())
        self.tabWidget.setCurrentIndex(0)
        self.progressBar_main.setValue(0)
        self.image = self.img_list[nrow]
        self.gender = self.gender_list[nrow]
        if self.gender[0] == 0:
            self.radioButton_female.setChecked(True)
        else:
            self.radioButton_male.setChecked(True)
        self.label_origin_show(self.image)
        self.mask = img_processing.mask(self.image)
        self.image_filter1 = img_processing.contrast(self.image, self.mask)
        self.image_filter2 = img_processing.equalization(self.image, self.mask)
        self.image_filter3 = img_processing.sobel(self.image, self.mask)
        self.roi = img_processing.roi_carpal_bone(self.image, self.mask)
        self.label_filter_show(self.image_filter1)
        self.label_roi_show(self.roi)
        self.output = self.bone_age_pred(self.gender, self.roi)
        self.label_prediction_show(self.output)
        self.pushButton_extract.setEnabled(True)
        self.progressBar_main.setValue(100)


if __name__ == "__main__":
    main_window = MainWindow()
    main_window.show()
    app.exec_()
