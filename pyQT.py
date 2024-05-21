import sys
import os
import json
import dlib
import cv2
from utils import get_face_chip, normalize_landmarks, read_landmarks, compare_landmarks_procrustes
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QPushButton

my_state = "<사용안함>"#사용안함, 예약중, 사용중

def resource_path(relative_path):  # ui 파일 가져오는 함수. 이 함수는 위키독스에서 가져온 것
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


# UI파일 연결
# 단, UI파일은 Python 코드 파일과 같은 디렉토리에 위치해야한다.
form_1 = resource_path('첫화면.ui')  #
form_class_1 = uic.loadUiType(form_1)[0]
# form_class_1 = uic.loadUiType("첫화면.ui")[0]

form_2 = resource_path('로그인.ui')
form_class_2 = uic.loadUiType(form_2)[0]
# form_class_2 = uic.loadUiType("로그인.ui")[0]

form_3 = resource_path('회원가입.ui')
form_class_3 = uic.loadUiType(form_3)[0]
# form_class_3 = uic.loadUiType("회원가입.ui")[0]

form_4 = resource_path('좌석예약.ui')
form_class_4 = uic.loadUiType(form_4)[0]


# form_class_4 = uic.loadUiType("좌석예약.ui")[0]

# 화면을 띄우는데 사용되는 Class 선언
class WindowClass(QMainWindow, form_class_1):
    # 시작화면 구성
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setupUi(self)

        # 버튼에 기능을 연결하는 코드
        self.login.clicked.disconnect()
        self.login.clicked.connect(self.system_login)
        self.join.clicked.disconnect()
        self.join.clicked.connect(self.system_join)

    # login 눌리면 작동할 함수
    def system_login(self):
        self.hide()  # 현재 화면 숨겨주고
        page1 = Page1(self)  # 페이지 1로 불러오고
        page1.show()  # 페이지 1를 보여준다(show())

    # join 눌리면 작동할 함수
    def system_join(self):
        self.hide()  # 현재 화면 숨겨주고
        print(1)
        page2 = Page2(self)  # 페이지 2로 불러오고
        page2.show()  # 페이지 2를 보여준다(show())


class Page1(QMainWindow, form_class_2):  # 로그인 화면
    def __init__(self, parent):
        super().__init__(parent)
        self.setupUi(self)
        self.login_button.clicked.disconnect()
        self.login_button.clicked.connect(self.complete_login)

        self.home_button.clicked.connect(self.home)
        self.home_button.setStyleSheet('border-image:url(C:/facial-landmarks-recognition/home.png);border:0px;')


    # home화면으로 이동
    def home(self):
        self.hide()  # 현재 화면 숨겨주고
        windowClass = WindowClass(self)  # 첫화면 불러오고
        windowClass.show()  # 첫화면 보여준다(show())


    # 확인 클릭 시 작동할 함수 -> 좌석예약(page3) 화면으로 이동
    def complete_login(self):
        id2 = self.id_text_edit2.toPlainText()
        pw2 = self.pw_text_edit2.toPlainText()

        # print("ID2:", id)
        # print("Password2:", pw)

        folder_path = os.path.join("C:/facial-landmarks-recognition/Capdi", id2)

        # 파일에서 사용자 정보 읽어오기

        if os.path.exists(folder_path):
            file_path = os.path.join(folder_path, "data.txt")
            with open(file_path, "r", encoding="utf-8") as file:
                user_info = json.load(file)
                saved_pw = user_info.get("pw")
                # 비밀번호 비교
                if pw2 == saved_pw:
                    self.hide()  # 현재 화면 숨겨주고
                    page3 = Page3(self)  # 페이지 3로 이동
                    page3.show()  # 페이지 3를 보여준다(show())
                else:
                    QMessageBox.warning(self, "로그인 오류", "아이디 또는 비밀번호가 잘못되었습니다.")
        else:
            QMessageBox.warning(self, "로그인 오류", "아이디 또는 비밀번호가 잘못되었습니다.")

        # if id2 == id and pw2 == pw:
        #     self.hide()  # 현재 화면 숨겨주고
        #     page3 = Page3(self)  # 페이지 2로 불러오고
        #     page3.show()  # 페이지 2를 보여준다(show())
        # else:
        #     QMessageBox.warning(self, "로그인 오류", "아이디 또는 비밀번호가 잘못되었습니다.")

class Page2(QMainWindow, form_class_3):  # 회원가입 화면
    def __init__(self, parent):
        super().__init__(parent)
        self.setupUi(self)
        self.join_button.clicked.disconnect()
        self.join_button.clicked.connect(self.complete_join)
        self.home_button.clicked.connect(self.home)
        self.home_button.setStyleSheet('border-image:url(home.png);border:0px;')

        self.student_id = None  # 학번을 저장할 변수

    def home(self):
        self.hide()
        windowClass = WindowClass(self)
        windowClass.show()

    def complete_join(self):
        """회원가입 완료 처리 함수"""
        global id
        global pw
        self.student_id = self.id_text_edit.toPlainText()
        name = self.name_text_edit.toPlainText()
        pw = self.pw_text_edit.toPlainText()
        folder_path = os.path.join("C:/facial-landmarks-recognition/Capdi", self.student_id)
        filename = os.path.join(folder_path, "landmark.txt")  # 랜드마크 파일 경로 지정

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
                self.register_face()  # 얼굴 등록 진행

    def register_face(self):
        folder_path = os.path.join("Capdi", self.student_id)
        filename = os.path.join(folder_path, "landmark.txt")

        QMessageBox.warning(self, "알림","카메라를 응시해주세요")

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
            
            if len(faces) > 0:
                for face in faces:
                    landmarks = predictor(img, face)
                    face_chip = get_face_chip(img, landmarks)
                    new_faces = detector(face_chip, 1)

                    for new_face in new_faces:
                        new_landmarks = predictor(face_chip, new_face)
                        face_landmarks = [(new_landmarks.part(n).x, new_landmarks.part(n).y) for n in range(68)]
                    normalized_landmarks = normalize_landmarks(face_landmarks, face_chip.shape)

                with open(filename, 'w') as f:
                    for (x, y) in normalized_landmarks:
                        f.write(f"{x},{y}\n")
                QMessageBox.warning(self, "알림", "회원가입 완료")
                break

        cap.release()
        cv2.destroyAllWindows()
        self.complete_join_final_step()  # 얼굴 등록 후 가입 완료 처리

    def complete_join_final_step(self):
        """회원가입 완료 후 최종 단계"""
        self.hide()
        self.parent().show()
        page1 = Page1(self)
        page1.show()
        
    # home화면으로 이동
    def home(self):
        self.hide()  # 현재 화면 숨겨주고
        windowClass = WindowClass(self)  # 첫화면 불러오고
        windowClass.show()  # 첫화면 보여준다(show())



class Page3(QMainWindow, form_class_4):  # 좌석예약 화면
    def __init__(self, parent):
        super().__init__(parent)
        self.setupUi(self)

        self.home_button.clicked.connect(self.home)
        self.home_button.setStyleSheet('border-image:url(home.png);border:0px;')


        # 예약시간
        self.hour1.clicked.connect(self.SelectTime)
        self.hour2.clicked.connect(self.SelectTime)
        self.hour3.clicked.connect(self.SelectTime)
        self.hour4.clicked.connect(self.SelectTime)
        self.hour5.clicked.connect(self.SelectTime)
        self.hour6.clicked.connect(self.SelectTime)

        # 좌석선택
        self.seat1.clicked.connect(self.SelectSeat)
        self.seat2.clicked.connect(self.SelectSeat)
        self.seat3.clicked.connect(self.SelectSeat)
        self.seat4.clicked.connect(self.SelectSeat)
        self.seat5.clicked.connect(self.SelectSeat)
        self.seat6.clicked.connect(self.SelectSeat)

        #촬영버튼
        self.photo.clicked.connect(self.complete_select)

    # home화면으로 이동
    def home(self):
        self.hide()  # 현재 화면 숨겨주고
        windowClass = WindowClass(self)  # 첫화면 불러오고
        windowClass.show()  # 첫화면 보여준다(show())


    def complete_select(self):
        global my_state
        my_state = "<예약중>" +str(seat_num) + "번 좌석 :: " + str(reserved_time) + "시간"
        print(my_state)
        print("reserved_time:", reserved_time)
        print("seat_num:", seat_num)
        self.hide()  # 현재 화면 숨겨주고
        #여기서부터 사진 비교 코드

    def SelectTime(self):  # 함수 내용 수정하기!!!
        global reserved_time
        if self.hour1.isChecked():
            reserved_time = 1
            print("GroupBox_rad1 Chekced")
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

    def SelectSeat(self):  # 함수 내용 수정하기!!!
        print(1)
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
                self.able.setStyleSheet("")  # 배경색을 원래대로 되돌리기 위해 빈 문자열을 설정합니다.
                self.unable.setStyleSheet("background-color: yellow;")
            print("GroupBox_rad1 Chekced")
        elif sender == self.seat2:
            seat_num = 2
            if seat2_state == True:
                print("true")
                self.able.setStyleSheet("background-color: yellow;")
                self.unable.setStyleSheet("")
            else:
                self.able.setStyleSheet("")  # 배경색을 원래대로 되돌리기 위해 빈 문자열을 설정합니다.
                self.unable.setStyleSheet("background-color: yellow;")
            print("GroupBox_rad2 Checked")
        elif sender == self.seat3:
            seat_num = 3
            if seat3_state == True:
                print("true")
                self.able.setStyleSheet("background-color: yellow;")
                self.unable.setStyleSheet("")
            else:
                self.able.setStyleSheet("")  # 배경색을 원래대로 되돌리기 위해 빈 문자열을 설정합니다.
                self.unable.setStyleSheet("background-color: yellow;")
            print("GroupBox_rad3 Checked")
        elif sender == self.seat4:
            seat_num = 4
            if seat4_state == True:
                print("true")
                self.able.setStyleSheet("background-color: yellow;")
                self.unable.setStyleSheet("")
            else:
                self.able.setStyleSheet("")  # 배경색을 원래대로 되돌리기 위해 빈 문자열을 설정합니다.
                self.unable.setStyleSheet("background-color: yellow;")
            print("GroupBox_rad4 Checked")
        elif sender == self.seat5:
            seat_num = 5
            if seat5_state == True:
                print("true")
                self.able.setStyleSheet("background-color: yellow;")
                self.unable.setStyleSheet("")
            else:
                self.able.setStyleSheet("")  # 배경색을 원래대로 되돌리기 위해 빈 문자열을 설정합니다.
                self.unable.setStyleSheet("background-color: yellow;")
            print("GroupBox_rad4 Checked")
        elif sender == self.seat6:
            seat_num = 6
            if seat6_state == True:
                print("true")
                self.able.setStyleSheet("background-color: yellow;")
                self.unable.setStyleSheet("")
            else:
                self.able.setStyleSheet("")  # 배경색을 원래대로 되돌리기 위해 빈 문자열을 설정합니다.
                self.unable.setStyleSheet("background-color: yellow;")
            print("GroupBox_rad4 Checked")


if __name__ == "__main__":
    # QApplication : 프로그램을 실행시켜주는 클래스
    app = QApplication(sys.argv)

    # WindowClass의 인스턴스 생성
    myWindow = WindowClass()

    # 프로그램 화면을 보여주는 코드
    myWindow.show()

    # 프로그램을 이벤트루프로 진입시키는(프로그램을 작동시키는) 코드
    app.exec()