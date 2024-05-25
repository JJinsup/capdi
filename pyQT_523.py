import sys
import os
import json
import dlib
import cv2
from utils import get_face_chip, normalize_landmarks, read_landmarks, compare_landmarks_procrustes
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox

my_state = "<사용안함>"  # 사용안함, 예약중, 사용중

def resource_path(relative_path):  # ui 파일 가져오는 함수. 이 함수는 위키독스에서 가져온 것
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

# UI 파일 연결
form_1 = resource_path('첫화면.ui')
form_class_1 = uic.loadUiType(form_1)[0]
form_2 = resource_path('로그인.ui')
form_class_2 = uic.loadUiType(form_2)[0]
form_3 = resource_path('회원가입.ui')
form_class_3 = uic.loadUiType(form_3)[0]
form_4 = resource_path('좌석예약.ui')
form_class_4 = uic.loadUiType(form_4)[0]

class WindowClass(QMainWindow, form_class_1):
    # 시작화면 구성
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.login.clicked.disconnect()
        self.login.clicked.connect(self.system_login)
        self.join.clicked.disconnect()
        self.join.clicked.connect(self.system_join)

    def system_login(self):
        self.hide()
        page1 = Page1(self)
        page1.show()

    def system_join(self):
        self.hide()
        page2 = Page2(self)
        page2.show()

class Page1(QMainWindow, form_class_2):
    def __init__(self, parent):
        super().__init__(parent)
        self.setupUi(self)
        self.login_button.clicked.disconnect()
        self.login_button.clicked.connect(self.complete_login)
        self.home_button.clicked.connect(self.home)
        self.home_button.setStyleSheet('border-image:url(C:/facial-landmarks-recognition/home.png);border:0px;')

    def home(self):
        self.hide()
        windowClass = WindowClass(self)
        windowClass.show()

    def complete_login(self):
        id2 = self.id_text_edit2.toPlainText()
        pw2 = self.pw_text_edit2.toPlainText()
        folder_path = os.path.join("C:/facial-landmarks-recognition/Capdi", id2)

        if os.path.exists(folder_path):
            file_path = os.path.join(folder_path, "data.txt")
            with open(file_path, "r", encoding="utf-8") as file:
                user_info = json.load(file)
                saved_pw = user_info.get("pw")
                if pw2 == saved_pw:
                    self.hide()
                    page3 = Page3(self, id2)
                    page3.show()
                else:
                    QMessageBox.warning(self, "로그인 오류", "아이디 또는 비밀번호가 잘못되었습니다.")
        else:
            QMessageBox.warning(self, "로그인 오류", "아이디 또는 비밀번호가 잘못되었습니다.")

class Page2(QMainWindow, form_class_3):
    def __init__(self, parent):
        super().__init__(parent)
        self.setupUi(self)
        self.join_button.clicked.disconnect()
        self.join_button.clicked.connect(self.complete_join)
        self.home_button.clicked.connect(self.home)
        self.home_button.setStyleSheet('border-image:url(home.png);border:0px;')
        self.student_id = None

    def home(self):
        self.hide()
        windowClass = WindowClass(self)
        windowClass.show()

    def complete_join(self):
        global id
        global pw
        self.student_id = self.id_text_edit.toPlainText()
        name = self.name_text_edit.toPlainText()
        pw = self.pw_text_edit.toPlainText()
        folder_path = os.path.join("C:/facial-landmarks-recognition/Capdi", self.student_id)
        filename = os.path.join(folder_path, "landmark.txt")

        if os.path.exists(filename):
            QMessageBox.warning(self, "알림", "이미 가입된 회원 입니다")
            self.close()
            page1 = Page1(self)
            page1.show()
        else:
            os.makedirs(folder_path)
            file_path = os.path.join(folder_path, "data.txt")
            user_info = {"name": name, "id": self.student_id, "pw": pw}
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(user_info, file, ensure_ascii=False)
            reply = QMessageBox.question(self, "알림", "얼굴 등록을 진행하시겠습니까?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.register_face()

    def register_face(self):
        folder_path = os.path.join("Capdi", self.student_id)
        filename = os.path.join(folder_path, "landmark.txt")

        QMessageBox.warning(self, "알림", "카메라를 응시해주세요")

        p = "shape_predictor_68_face_landmarks.dat"
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(p)

        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from camera. Exiting...")
                break

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(img, 1)
            face_landmarks = []

            if len(faces) > 0:
                for face in faces:
                    landmarks = predictor(img, face)
                    face_chip = get_face_chip(img, landmarks)
                    new_faces = detector(face_chip, 1)

                    for new_face in new_faces:
                        new_landmarks = predictor(face_chip, new_face)
                        face_landmarks = [(new_landmarks.part(n).x, new_landmarks.part(n).y) for n in range(68)]
                    normalized_landmarks = normalize_landmarks(face_landmarks, face_chip.shape)

                if normalized_landmarks:
                    with open(filename, 'w') as f:
                        for (x, y) in normalized_landmarks:
                            f.write(f"{x},{y}\n")
                    QMessageBox.warning(self, "알림", "회원가입 완료")
                    break
                else:
                    print("Failed to normalize landmarks. Retrying...")

        cap.release()
        cv2.destroyAllWindows()
        self.complete_join_final_step()

    def complete_join_final_step(self):
        self.hide()
        self.parent().show()
        page1 = Page1(self)
        page1.show()

class Page3(QMainWindow, form_class_4):
    def __init__(self, parent, student_id):
        super().__init__(parent)
        self.setupUi(self)
        self.student_id = student_id

        self.home_button.clicked.connect(self.home)
        self.home_button.setStyleSheet('border-image:url(home.png);border:0px;')

        self.hour1.clicked.connect(self.SelectTime)
        self.hour2.clicked.connect(self.SelectTime)
        self.hour3.clicked.connect(self.SelectTime)
        self.hour4.clicked.connect(self.SelectTime)
        self.hour5.clicked.connect(self.SelectTime)
        self.hour6.clicked.connect(self.SelectTime)

        self.seat1.clicked.connect(self.SelectSeat)
        self.seat2.clicked.connect(self.SelectSeat)
        self.seat3.clicked.connect(self.SelectSeat)
        self.seat4.clicked.connect(self.SelectSeat)
        self.seat5.clicked.connect(self.SelectSeat)
        self.seat6.clicked.connect(self.SelectSeat)

        self.photo.clicked.connect(self.complete_select)

    def home(self):
        self.hide()
        windowClass = WindowClass(self)
        windowClass.show()

    def complete_select(self):
        global my_state
        landmarks_list = []
        my_state = "<예약중>" + str(seat_num) + "번 좌석 :: " + str(reserved_time) + "시간"
        print(my_state)
        print("reserved_time:", reserved_time)
        print("seat_num:", seat_num)
        self.compare_landmark()

    def compare_landmark(self):
        folder_path = os.path.join("C:/facial-landmarks-recognition/Capdi", self.student_id)
        filename = os.path.join(folder_path, "landmark.txt")

        QMessageBox.warning(self, "알림", "카메라를 응시해주세요")

        p = "shape_predictor_68_face_landmarks.dat"
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(p)
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from camera. Exiting...")
                break

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(img, 1)
            face_landmarks = []
            landmarks1 = None

            if len(faces) > 0:
                for face in faces:
                    landmarks = predictor(img, face)
                    face_chip = get_face_chip(img, landmarks)
                    new_faces = detector(face_chip, 1)

                    for new_face in new_faces:
                        new_landmarks = predictor(face_chip, new_face)

                        for n in range(0, 68):
                            x = new_landmarks.part(n).x
                            y = new_landmarks.part(n).y
                            face_landmarks.append((x, y))
                    landmarks1 = normalize_landmarks(face_landmarks, face_chip.shape)

                if landmarks1:
                    break
                else:
                    print("Failed to normalize landmarks. Retrying...")
            else:
                QMessageBox.warning(self, "알림", "실패")

        if landmarks1:
            landmarks2 = read_landmarks(filename)
            disparity, result = compare_landmarks_procrustes(landmarks1, landmarks2, 0.005)

            if result:
                QMessageBox.warning(self, "알림", f"인증에 성공했습니다. (차이: {disparity:.4f})")
                self.hide()
                page1 = Page1(self)
                page1.show()
            else:
                QMessageBox.warning(self, "알림", f"인증에 실패했습니다. (차이: {disparity:.4f})")
                self.hide()
                page3 = Page3(self, self.student_id)
                page3.show()
        else:
            QMessageBox.warning(self, "알림", "랜드마크 정규화에 실패했습니다. 다시 시도해주세요.")

        cap.release()
        cv2.destroyAllWindows()

    def SelectTime(self):
        global reserved_time
        if self.hour1.isChecked():
            reserved_time = 1
            print("GroupBox_rad1 Checked")
        elif self.hour2.isChecked():
            reserved_time = 2
            print("GroupBox_rad2 Checked")
        elif self.hour3.isChecked():
            reserved_time = 3
            print("GroupBox_rad3 Checked")
        elif self.hour4.isChecked():
            reserved_time = 4
            print("GroupBox_rad4 Checked")
        elif self.hour5.isChecked():
            reserved_time = 5
            print("GroupBox_rad5 Checked")
        elif self.hour6.isChecked():
            reserved_time = 6
            print("GroupBox_rad6 Checked")

    def SelectSeat(self):
        global seat_num
        global seat1_state
        global seat2_state
        global seat3_state
        global seat4_state
        global seat5_state
        global seat6_state
        seat1_state = True
        seat2_state = True
        seat3_state = True
        seat4_state = True
        seat5_state = True
        seat6_state = True
        sender = self.sender()
        if sender == self.seat1:
            seat_num = 1
            if seat1_state == True:
                print("true")
                self.able.setStyleSheet("background-color: yellow;")
                self.unable.setStyleSheet("")
            else:
                self.able.setStyleSheet("")
                self.unable.setStyleSheet("background-color: yellow;")
            print("GroupBox_rad1 Checked")
        elif sender == self.seat2:
            seat_num = 2
            if seat2_state == True:
                print("true")
                self.able.setStyleSheet("background-color: yellow;")
                self.unable.setStyleSheet("")
            else:
                self.able.setStyleSheet("")
                self.unable.setStyleSheet("background-color: yellow;")
            print("GroupBox_rad2 Checked")
        elif sender == self.seat3:
            seat_num = 3
            if seat3_state == True:
                print("true")
                self.able.setStyleSheet("background-color: yellow;")
                self.unable.setStyleSheet("")
            else:
                self.able.setStyleSheet("")
                self.unable.setStyleSheet("background-color: yellow;")
            print("GroupBox_rad3 Checked")
        elif sender == self.seat4:
            seat_num = 4
            if seat4_state == True:
                print("true")
                self.able.setStyleSheet("background-color: yellow;")
                self.unable.setStyleSheet("")
            else:
                self.able.setStyleSheet("")
                self.unable.setStyleSheet("background-color: yellow;")
            print("GroupBox_rad4 Checked")
        elif sender == self.seat5:
            seat_num = 5
            if seat5_state == True:
                print("true")
                self.able.setStyleSheet("background-color: yellow;")
                self.unable.setStyleSheet("")
            else:
                self.able.setStyleSheet("")
                self.unable.setStyleSheet("background-color: yellow;")
            print("GroupBox_rad5 Checked")
        elif sender == self.seat6:
            seat_num = 6
            if seat6_state == True:
                print("true")
                self.able.setStyleSheet("background-color: yellow;")
                self.unable.setStyleSheet("")
            else:
                self.able.setStyleSheet("")
                self.unable.setStyleSheet("background-color: yellow;")
            print("GroupBox_rad6 Checked")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec()
