# 얼굴인식 프로젝트

## Step1 - 얼굴 사진 100장 찍어 데이터 수집하기
- CascadeClassifier는, 단순히 얼굴 검출기라기보다는, 유사-하르 필터를 이용하여 유사도로 특정 객체를 검출해내는 검출기로, 어떤 객체를 검출할지를 바로 이 xml파일에 설정함으로써 변경 가능하고, load 메소드를 통해 미리 훈련된 분류기 정보를 가져올수 있습니다.
- 미리 훈련된 분류기 XML 파일은 OpenCV에서 제공

```python
import cv2
import numpy as np
import os

  

# 유사-하르 필터를 이용한 얼굴 검출기 CascadeClassifier을 사용하여 분류기를 face_classifier 로 지정하기

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

  

# 전체 사진에서 얼굴 부위만 추출하는 함수
def face_extractor(img):

    # 흑백처리
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 얼굴 찾기
    faces = face_classifier.detectMultiScale(
        gray,
        scaleFactor=1.1,     # 더 세밀하게 얼굴 탐지
        minNeighbors=4,      # 노이즈 줄이기
        minSize=(80, 80)     # 너무 작은 얼굴은 무시
    )

    if len(faces) == 0:
        return None

    for (x, y, w, h) in faces:
        # 해당 얼굴 크기만큼 cropped_face에 잘라 넣기
        # 근데... 얼굴이 2개 이상 감지되면?? 가장 마지막의 얼굴만 남을 듯
        cropped_face = img[y:y + h, x:x + w]
    return cropped_face

# 저장 폴더 생성
save_dir = 'faces'
os.makedirs(save_dir, exist_ok=True)

# 카메라 실행
cap = cv2.VideoCapture(0)
# 저장할 이미지 카운트 변수
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라 프레임을 읽을 수 없습니다.")
        break
    # 얼굴 감지 하여 얼굴만 가져오기
    face_img = face_extractor(frame)
    if face_img is not None:
        count += 1
        # 얼굴 이미지 크기를 200x200으로 조정
        face = cv2.resize(face_extractor(frame), (200, 200))
        # 조정된 이미지를 흑백으로 변환
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # ex > faces/user0.jpg   faces/user1.jpg ....
        file_path = os.path.join(save_dir, f'user{count}.jpg')
        cv2.imwrite(file_path, face)  # faces폴더에 jpg파일로 저장

        # 화면에 얼굴과 현재 저장 개수 표시
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Cropper', face)
    else:
        print("Face not Found")
  
    if cv2.waitKey(10) == ord('q') or count == 100:
        break
  
cap.release()
cv2.destroyAllWindows()
print('Colleting Samples Complete!!!')
```

## Step2 - 100장의 사진을 학습 시키기


- LBPH (Local Binary Patterns Histograms) : 전통적인 머신러닝 기반 모델
    - grayscale 이미지만 사용
    - 간단한 수치 비교, 패턴 분석
    - 조명, 각도, 표정 변화에 약함

- 필요한 라이브러리

```python
    Python 3.7-3.11
    pip install opencv-contrib-python==4.8.1.78
    pip install numpy==1.26.4
```

```python
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

data_path = 'faces/'

#faces폴더에 있는 파일 리스트 얻기
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

#데이터와 매칭될 라벨 변수
Training_Data, Labels = [], []

#파일 개수 만큼 반복하기
for i, files in enumerate(onlyfiles):    
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #이미지 파일이 아니거나 못 읽어 왔다면 무시
    if images is None:
        continue    
    #Training_Data 리스트에 이미지를 바이트 배열로 추가
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    #Labels 리스트엔 카운트 번호 추가
    Labels.append(i)

# 학습 데이터가 없을 경우 종료
if len(Training_Data) == 0:
    print("There is no data to train.")
    exit()

#Labels를 32비트 정수로 변환
Labels = np.asarray(Labels, dtype=np.int32)

# # 문자열 라벨을 숫자 인코딩 (ex: user1 → 0, user2 → 1)
# label_encoder = LabelEncoder()
# Labels = label_encoder.fit_transform(Label_Names)
  
# LBPH 얼굴 인식 모델 생성 (opencv-contrib-python 필요)
model = cv2.face_LBPHFaceRecognizer.create()

#학습 시작
model.train(np.asarray(Training_Data), np.asarray(Labels))
print("모델 훈련이 완료되었습니다!!!!!")

# 모델 및 라벨 인코더 저장
model.save('trained_model.yml')

#joblib.dump(label_encoder, 'label_encoder.pkl')
print("모델 및 라벨 인코더 저장 완료!")
```
## Step3 - 얼굴 인식해서 동일인인지 구분하기

```python
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

# 학습된 모델 불러오기
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def face_detector(img, size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return img, []
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi

# 먼저 model을 로드해야 함
try:
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read('trained_model.yml')
except Exception as e:
    print(f"모델 로드 실패: {e}")
    exit()

# 카메라 열기
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # 얼굴 검출 시도
    image, face = face_detector(frame)
    try:
        if len(face) != 0:  # face 배열이 비어있지 않은 경우에만
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            # 위에서 학습한 모델로 예측하기
            result = model.predict(face)
            # result[1]은 신뢰도 값으로 0에 가까울수록 자신과 같다는 의미로 신뢰도가 높음
            if result[1] < 50:
                confidence = int(100*(1-(result[1])/300))  # 신뢰도를 %로 나타냄
                display_string = str(confidence)+'% Confidence it is user'  # 유사도 화면에 표시
                cv2.putText(image, display_string, (100, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 120, 255), 2)
            # 75 보다 크면 동일 인물로 간주해 접근허가!
            if confidence > 75:
                cv2.putText(image, "ACCESS GRANTED", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            else:
               # 75 이하면 타인.. 접근불가!!
                cv2.putText(image, "ACCESS DENIED", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(image, "Face Not Found", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    except Exception as e:
        print(f"처리 중 에러: {e}")
        cv2.putText(image, "Face Not Found", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    # 화면 표시를 while 루프 안에서 한 번만 실행
    cv2.imshow('Face Cropper', image)
    # 키 입력 체크
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

