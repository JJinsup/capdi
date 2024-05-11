# capdi
자료공유

https://github.com/italojs/facial-landmarks-recognition 에서 main.py 수정해서 진행중

shape_predictor_68_face_landmarks.dat기반 랜드마크 추출이고
txt파일에 x,y좌표 저장됨.


[landmark_compare.py]
카메라 영상 1프레임만 따서 랜드마크 추출하고 이미 저장된 txt파일 좌표랑 distance비교해서 TRUE FALSE반환하는데
내 사진이랑 실시간 영상 따서 추출한 랜드마크랑 비교했을 때 FALSE만 떠서 이 방법은 힘들 것 같음.
=> 랜드마크 추출하기 전에 normalization하고 해보려고 하고 있음
Normalization in Face Recognition with Dlib Facial Landmarks 참고해서

[landmark_img.py]
이미지에서 랜드마크 따서 저장해줌

<2024.05.11>

[landmark_cropped.py]
이미지에 대해 얼굴부분만 남기고 나머지는 다 잘라버린 다음 랜드마크 추출한다.

[landmark_cropped_compare.py]
원래 비교 방법이랑 똑같은데 frame에다가 cropped처럼 전처리 하고 랜드마크 추출해서 저장한 다음 비교함.
threshold25까지는 내 사진이랑 TRUE. 25미만은 FALSE나옴...(다른 사진이랑은 threshold 40?까지 FALSE나옴)

참고자료
https://github.com/khanetor/face-alignment-dlib에서 app.py
https://blog.naver.com/chandong83/221488347537(dlib관련 좋은 사용법 많음)
https://ukayzm.github.io/unknown-face-classifier/(distance비교하는 방식이 좀 달라서 낼 해볼게)
https://wnsgur0329.tistory.com/21(encoding)
