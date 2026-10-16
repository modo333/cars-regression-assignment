# Car Stopping Distance — Regression & Neural Network

> 공개 자동차 데이터를 이용한 선형회귀 · 2차 다항회귀 · 인공신경망 비교 프로젝트  
> 데이터처리개론 | Python · scikit-learn · GitHub Actions

## 프로젝트 개요

자동차 속도(`speed`)와 정지거리(`dist`)의 관계를 세 가지 모델로 분석합니다. R의 공개 `cars` 데이터 50개를 사용해 학습 적합도와 별도 테스트 점수를 구분합니다. [브라우저에서 직접 실행하기](https://cars-regression-lab-yunseok.hushedseal256.chatgpt.site)에서도 데이터 수정과 예측이 가능합니다.

| 항목 | 내용 |
|:--|:--|
| 데이터 | R `cars`, 관측치 50개 |
| 입력 / 출력 | 속도 `speed` (mph) → 정지거리 `dist` (ft) |
| 모델 | 선형회귀 / 2차 다항회귀 / 1–8–1 인공신경망 |
| 평가지표 | 전체 50개 학습 R² + 40개 학습·10개 테스트 R² |
| 산출물 | 회귀식, R², 비교 그래프 PNG |

## 분석 결과

| 모델 | 전체 50개 학습 R² | 40/10 분할 테스트 R² |
|:--|--:|--:|
| 선형 | 0.651079 | **0.632373** |
| 2차 | 0.667331 | 0.610155 |
| 인공신경망 | **0.692287** | 0.485647 |

기존 선형·2차 결과는 그대로 재현됩니다. 신경망의 학습 R²는 가장 높지만, 따로 남긴 10개 데이터에 대한 R²는 더 낮습니다. **복잡한 모델이 반드시 더 잘 예측하지는 않습니다.** 테스트는 원본 순서의 5·10·…·50번째 행을 분리한 작은 표본이므로, 이 점수 하나만으로 우열을 확정할 수는 없습니다.

![실제 관측값과 세 모델 비교 그래프](cars_regression.png)

## 실행 방법

### GitHub Actions에서 실행

1. 저장소 상단 **Actions** → 왼쪽 **Run regression**을 엽니다.
2. **Run workflow** → 초록색 **Run workflow**를 누릅니다.
3. 실행 완료 후 **Run regression** 단계에서 세 모델의 R²와 속도별 예측값을 확인합니다.
4. 실행 상세의 **Artifacts → regression-graph**에서 생성된 그래프를 내려받습니다.

`regression_cars.py`, `neural_network.py`, `cars.csv`, `requirements.txt` 또는 워크플로가 변경돼도 자동 실행됩니다.

### 로컬에서 실행

Python 3.11 이상 환경에서:

```bash
python -m pip install -r requirements.txt
python regression_cars.py --speed 20
```

같은 폴더의 `cars.csv`를 읽어 세 모델의 학습·테스트 R² 및 예측값을 출력하고, `cars_regression.png`를 저장합니다. `--speed`로 예측 속도를 바꿀 수 있습니다.

## 처리 흐름

`CSV 로드 → 40/10 분할 → 선형·이차·신경망 학습과 테스트 → 전체 자료 재학습 → 예측과 시각화`

- **선형 모델:** `dist = b₀ + b₁·speed`
- **2차 모델:** `dist = b₀ + b₁·speed + b₂·speed²`
- **인공신경망:** 입력 1개 → tanh 은닉 뉴런 8개 → 연속값 출력 1개. 학습 자료로 속도·거리 표준화 후 오차역전파와 Adam(2,500회, 학습률 0.01, L2 계수 0.001)으로 가중치를 조정합니다. 초기 가중치를 고정해 실행 결과를 재현합니다.
- 선형·2차 모델은 **오차 제곱합을 최소화**합니다. 2차 모델은 계수에 대해서는 선형이지만, 속도와 거리의 관계는 곡선입니다.
- 테스트 평가에서는 세 모델 모두 **같은** 40개로 학습하고 남긴 10개에서 R²를 계산합니다. 예측과 그래프는 50개 전체로 다시 학습한 모델입니다.

## 저장소 구성

```text
├── .github/workflows/run-regression.yml  # 수동·자동 실행
├── cars.csv                              # 공개 데이터 50개
├── cars_regression.png                   # 결과 그래프 예시
├── regression_cars.py                    # 분석 프로그램
├── neural_network.py                     # 학습 가능한 인공신경망
├── requirements.txt                      # Python 의존성
└── README.md                             # 프로젝트 설명
```

## 데이터 출처 및 주의사항

- [R 공식 `cars` 설명](https://search.r-project.org/R/refmans/datasets/html/cars.html)
- [공개 CSV 원본](https://vincentarelbundock.github.io/Rdatasets/csv/datasets/cars.csv)

이 자료는 1920년대 자동차 기록입니다. 현대 자동차의 실제 안전 제동거리 산정에 사용하지 마세요.

신경망 구조와 표준화의 배경은 [scikit-learn 신경망 설명](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)을 참고했습니다. 이 프로젝트의 신경망 학습 코드는 해당 라이브러리 함수를 호출하지 않고 NumPy로 직접 구현했습니다.
