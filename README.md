# iot-dataanalysis-2025
2025년 IoT 개발자 과정 빅데이터분석 리포지토리

# 1일차 

### 머신러닝/딥러닝

<img src="./image/ml0001.png" width="750">
(출처 : NVIDIA)

- 인공지능(Artificial Intelligence: AI)의 분야
    - 컴퓨터가 사람의 행동을 흉내내는 모든 기술

- 머신러닝
    - 인공지능 하위 집합
    - 통계적 방법을 이용, 기계를 학습시키는 인공지능 기술

- 딥러닝
    - 머신러닝의 하위 집합
    - 신경망 기술을 이용, 머신러닝 기술 중 하나

- 인공지능 역사
    - 1943 - MCP뉴런 이론
    - 1950 - 튜링(앨런 튜링) 테스트, 인공지능 테스트
    - 1957 - 퍼셉트론 이론
    - 1974 - 1차 AI겨울. 컴퓨터 성능 한계
    - 1980 - AI붐. 전문가 시스템(머신러닝)
    - 1987 - 2차 AI겨울. 전문가 시스템 실패
    - 2010 - 컴퓨팅 HW환경 비약적 발전. AI희망
    - 2015 - 템서플로 발표
    - 2016 - 알파고

### 개발환경

#### 코랩
- Google Colaboratory
- 구글에서 만든 온라인 주피터노트북 개발 플랫폼
- 구글 드라이브 연동, 구글 서버 하드웨어 사용
- 어디서나 파이썬 학습, 개발 등 가능
- https://colab.research.google.com/?hl=ko
- 런타임 유형
    - CPU, T4 GPU, v2-8 TPU - 무료
    - A100 GPU, L4 GPU, v5e-1 TPY- 유료
- 무료에서는 80분 넘어서면 세션이 끊어짐

#### VSCode
- 로컬 직접 환경 설정
- 사이킷런, 텐서플로, 쿠다, 파이토치...

##### 파이썬 가상환경
- 가상환경 생성 명령어
```shell
> python -m venv mlvenv
```
- 가상환경 사용
```shell
> .\mlvenv\Scripts\activate
(mlvenv) PS C:\Source\iot-dataanalysis-2025>
```

- 가상환경은 깃허브에 올라가지 않도록 처리
- .gitignore에 /mlvenv 추가 후 깃허브 우선 푸시

- 맷플롯립 설치
```shell
> pip install matplotlib
```

- 맷플롯립 한글 설정
```python
from matplotlib import rcParams, font_manager, rc

font_path = 'C:/Windows/Fonts/NanumGothic.ttf' # 나눔고딕코딩 사용, 나눔고딕에서 오류발생(!)
font = font_manager.FontProperties(fname=font_path).get_name() # 실제 설치된 폰트 이름조회
rc('font', family=font) # 한글깨짐현상 해결!!
rcParams['axes.unicode_minus'] = False # 한글 사용시 마이너스 표시 깨짐 해결!
```

- 사이킷런 설치
```shell
> pip install scikit-learn
```

- 텐서플로 설치
```shell
> pip install tensorflow==2.15.0
```

### 첫번째 머신러닝
- 캐글 생선 데이터
    - https://www.kaggle.com/datasets/vipullrathod/fish-market
    
- 길이를 보고 도미(bream)인지 빙어(smelt)인지 판별
- 이진 분류(Binary Classification)

## 8일차