# X-Age
Bone Age prediction program

자동 소아(1~10세) 골연령 측정 프로그램
--------------
## 개요
1. 자동 전처리 기능으로 뼈를 강조하고 골연령 판단에 중요한 ROI영역을 추출해 참고자료 제공
2. CNN을 통해 골연령을 예측
3. 기존 DB와 연결하여 사진을 업로드 하면 참고자료와 예측 골연령을 제공하는 PYQT기반 어플리케이션 개발
--------------
## 이미지 전처리

```
Note
- 사진들 간의 명도와 채도의 차이가 큼
- 손 외의 다른 물체들이 존재
```

### 전처리 Type 1
1. 마스크 생성(손 추출)
   - 메디안 블러, 침식연산, 이진화
   - 칸투어 계산 후 영역크기가 가장넓은 칸투어 추출
2. 뼈강조
   - TOPHAT모폴로지, 가우시안 블러, 이진화

### 전처리 Type 2
1. 마스크 생성
   - 이미지 밝기를 분리할수 있는 LAB채널로 변환
   - 메디안 블러, 이진화
   - 칸투어 계산 후 영역크기가 가장넓은 칸투어 추출
2. 마스크 기준으로 좌우 상하 영역제거
3. 뼈강조
   - contrast 함수
   - equalization
4. 회전
5. ROI
   - convexhull을 사용해서 기준이 되는 지점 찾기
   - 엄지-손목 경계점 기준으로 CarpalBone(손목뼈) ROI
   - 뼈강조

## CNN 모델 학습
```
CentOS Tesla v100 환경에서 GPU를 활용하여 학습속도
ImageNet에서 좋은 성능을 낸 모델과 gender dnn layer를 concatenate 하여 모델링
```
### Data
   - 국내 소아과 X-ray 싸진 (1~10세 남녀 총 441명)
   - RSNA Bone Age Contest at Kaggle(<https://www.kaggle.com/kmader/rsna-bone-age>)

### Train
   - 이미지와 Gender 두가지를 Input으로 사용
   - ImageNet에서 좋은 성과를 내었던 Xception, ResNet, Vgg를 기반으로 학습
   - 관련 논문중 tjnet을 참고하여 직접 모델링 하여 학습
   - 평가 지표 : MAE (0.34 year)
