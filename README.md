# logo-detector
YOLOv2를 사용한 브랜드 로고 검출 프로그램
![Image](https://github.com/user-attachments/assets/97a03654-debf-47aa-b9a3-4663b115cb2c)

#### 개발환경
CUDA 10.0, cuDNN, python3.7, Tensorflow 1.15.0, JAVA, eclipse

#### 팀원
3명

#### 학습률
약 88%

#### 실행화면
<img width="790" alt="Image" src="https://github.com/user-attachments/assets/e2fd4552-5827-4280-9b7b-fc4530fe1547" />

#### 주의사항
<details>
<summary>응용 프로그램 실행 관련</summary>

- 응용 프로그램 실행 시 Detection 폴더까지의 절대 경로를 java 소스코드에 입력해야 한다.
- 이미지 이름 및 폴더 이름에 띄어쓰기(공백)가 들어가면 안된다.
</details>

<details>
<summary>파일 경로 안내</summary>

- `...\Detection\data\face\test\images` : 응용 프로그램의 이미지 불러오기 기능에서 생성되는 이미지 저장
- `...\Detection\data\face\test\annotations` : 응용 프로그램의 이미지 불러오기 기능에서 생성되는 anno파일 저장
- `...\Detection\data\face\test\redraws\images` : 로고 검출된 이미지가 저장
</details>

<details>
<summary>브랜드 추가 학습 시 유의사항</summary>

추후 학습할 브랜드를 증가시킨다면 아래 파일들의 수정이 필요합니다:
- `Detection\data\face\test\classes.json`
- `Detection\data\face\train\classes.json`
- `train.py`의 num_classes 값
- `test.py`의 num_classes 값
</details>