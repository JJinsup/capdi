import sys
import os
import json
import dlib
import cv2
from utils import get_face_chip, normalize_landmarks, read_landmarks, compare_landmarks_cosine
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QLabel

def resource_path(relative_path):
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


form_1 = resource_path('첫화면.ui')
form_class_1 = uic.loadUiType(form_1)[0]
form_2 = resource_path('로그인.ui')
form_class_2 = uic.loadUiType(form_2)[0]
form_3 = resource_path('회원가입.ui')
form_class_3 = uic.loadUiType(form_3)[0]
form_4 = resource_path('좌석예약.ui')
form_class_4 = uic.loadUiType(form_4)[0]
form_5 = resource_path('얼굴인증.ui')
form_class_5 = uic.loadUiType(form_5)[0]
form_6 = resource_path('시간보여줌.ui')
form_class_6 = uic.loadUiType(form_6)[0]

class WindowClass(QMainWindow, form_class_1):
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
        self.home_button.setStyleSheet('border-image:url(home.png);border:0px;')

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
        folder_path = os.path.join("C:/facial-landmarks-recognition/Capdi", self.student_id)
        filename = os.path.join(folder_path, "landmark.txt")

        QMessageBox.warning(self, "알림", "카메라를 응시해주세요")

        p = "shape_predictor_68_face_landmarks.dat"
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(p)

        cap = cv2.VideoCapture(0)
        normalized_landmarks = None

        try:
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
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()

        self.complete_join_final_step()

    def complete_join_final_step(self):
        self.hide()
        page1 = Page1(self)
        page1.show()

class Page3(QMainWindow, form_class_4):
    def __init__(self, parent, student_id):
        super().__init__(parent)
        self.setupUi(self)
        self.student_id = student_id
        self.reserved_times = set()
        self.selected_seat = None
        self.reserved_seats = self.load_reserved_seats()  # 예약된 좌석을 불러옴

        self.home_button.clicked.connect(self.home)
        self.home_button.setStyleSheet('border-image:url(home.png);border:0px;')

        self.hour1.clicked.connect(lambda: self.SelectTime(1))
        self.hour2.clicked.connect(lambda: self.SelectTime(2))
        self.hour3.clicked.connect(lambda: self.SelectTime(3))
        self.hour4.clicked.connect(lambda: self.SelectTime(4))
        self.hour5.clicked.connect(lambda: self.SelectTime(5))
        self.hour6.clicked.connect(lambda: self.SelectTime(6))

        self.seat1.clicked.connect(lambda: self.SelectSeat(1))
        self.seat2.clicked.connect(lambda: self.SelectSeat(2))
        self.seat3.clicked.connect(lambda: self.SelectSeat(3))
        self.seat4.clicked.connect(lambda: self.SelectSeat(4))
        self.seat5.clicked.connect(lambda: self.SelectSeat(5))
        self.seat6.clicked.connect(lambda: self.SelectSeat(6))

        self.photo.clicked.connect(self.complete_select)

    def home(self):
        self.hide()
        windowClass = WindowClass(self)
        windowClass.show()

    def SelectTime(self, hour):
        self.reserved_times.add(hour)

    def SelectSeat(self, seat_number):
        if seat_number in self.reserved_seats:
            QMessageBox.warning(self, "Error", "이미 예약된 좌석입니다.")
            return

        if self.selected_seat == seat_number:
            return

        if self.selected_seat:
            # 이전에 선택한 좌석의 색상 리셋
            previous_seat_button = getattr(self, f"seat{self.selected_seat}")
            previous_seat_button.setStyleSheet("")

        # 새로 선택한 좌석의 색상 변경
        seat_button = getattr(self, f"seat{seat_number}")
        seat_button.setStyleSheet("background-color: #F2F5A9;")
        self.selected_seat = seat_number

    def complete_select(self):
        if not self.reserved_times:
            QMessageBox.warning(self, "Error", "Please select at least one time.")
            return
        if not self.selected_seat:
            QMessageBox.warning(self, "Error", "Please select a seat.")
            return

        self.save_selection()
        # QMessageBox.information(self, "Success", "Reservation completed.")
        QMessageBox.information(self, "Success", "자리에 가서 얼굴인증을 진행해주세요.")
        self.hide()  # 현재 페이지 숨기기
        page5 = Page5(self,self.student_id)  # form_5 페이지 열기
        page5.show()
        # self.close()  # 프로그램 종료


    def save_selection(self):
        folder_path = os.path.join("C:/facial-landmarks-recognition/Capdi", self.student_id)
        time_selection_path = os.path.join(folder_path, "time_selection.txt")
        seat_selection_path = os.path.join(folder_path, "seat_selection.txt")

        with open(time_selection_path, "w", encoding="utf-8") as file:
            json.dump(list(self.reserved_times), file, ensure_ascii=False)

        with open(seat_selection_path, "w", encoding="utf-8") as file:
            json.dump(self.selected_seat, file, ensure_ascii=False)

    def load_reserved_seats(self):
        reserved_seats = set()
        base_path = "C:/facial-landmarks-recognition/Capdi"

        for member_id in os.listdir(base_path):
            member_folder_path = os.path.join(base_path, member_id)
            seat_file_path = os.path.join(member_folder_path, "seat_selection.txt")

            if os.path.exists(seat_file_path):
                with open(seat_file_path, "r", encoding="utf-8") as file:
                    selected_seat = json.load(file)
                    reserved_seats.add(selected_seat)

        return reserved_seats


class Page5(QMainWindow, form_class_5):
    def __init__(self, parent, student_id):
        super().__init__(parent)
        self.setupUi(self)
        self.student_id=student_id
        self.facebutton.clicked.connect(self.system_face)
        # self.home_button.clicked.connect(self.home)
        # self.home_button.setStyleSheet('border-image:url(home.png);border:0px;')

    def system_face(self):
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
            disparity, result = compare_landmarks_cosine(landmarks1, landmarks2, 0.004)

            if result:
                QMessageBox.warning(self, "알림", f"인증에 성공했습니다. (차이: {disparity:.4f})")
                self.hide()
                page6 = Page6(self, self.student_id)
                page6.show()
            else:
                QMessageBox.warning(self, "알림", f"인증에 실패했습니다. (차이: {disparity:.4f})")
                self.hide()
                page5 = Page5(self, self.student_id)
                page5.show()
        else:
            QMessageBox.warning(self, "알림", "랜드마크 정규화에 실패했습니다. 다시 시도해주세요.")

        cap.release()
        cv2.destroyAllWindows()

    def home(self):
        self.hide()
        windowClass = WindowClass(self)
        windowClass.show()

class Page6(QMainWindow, form_class_6):
    def __init__(self, parent,student_id):
        super().__init__(parent)
        self.student_id=student_id
        self.setupUi(self)
        # self.facebutton.clicked.connect(self._compare_landmarks)
        # self.home_button.clicked.connect(self.home)
        # self.home_button.setStyleSheet('border-image:url(home.png);border:0px;')

    def home(self):
        self.hide()
        windowClass = WindowClass(self)
        windowClass.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    sys.exit(app.exec_())