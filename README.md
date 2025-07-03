# dacon-llm-challenge3

---

## 🧠 KoELECTRA 기반 생성 AI 텍스트 판별 모델 (0.79+ AUC)

이 프로젝트는 **KoELECTRA-small-v3-discriminator** 모델을 활용하여 **AI 생성 텍스트 여부를 분류**하는 분류기를 학습하고, 예측 결과를 제출 형식에 맞게 생성합니다.

---

### 📁 프로젝트 구조

```
📦open/
 ┣ 📄main.py                  # 실행 파일
 ┣ 📄train.csv               # 학습 데이터
 ┣ 📄test.csv                # 테스트 데이터
 ┣ 📄sample_submission.csv   # 제출 형식
 ┗ 📁results/                # 모델 결과 저장
```

---

### ✅ 실행 환경

* Python >= 3.8
* torch
* transformers
* pandas
* scikit-learn

설치 예시:

```bash
pip3 install torch transformers pandas scikit-learn
```

---

### 🚀 실행 방법

```bash
python3 main.py
```

---

### 🧩 주요 기능 설명

| 항목         | 설명                                          |
| ---------- | ------------------------------------------- |
| **모델**     | `monologg/koelectra-small-v3-discriminator` |
| **토큰화**    | HuggingFace `AutoTokenizer` 사용              |
| **모델 학습**  | HuggingFace `Trainer` 기반                    |
| **데이터 분할** | `train_test_split` (train/val = 9:1)        |
| **입력 길이**  | `max_length=256`                            |
| **출력**     | 소프트맥스 확률로 `generated` 컬럼 생성                 |
| **결과 저장**  | `submission_날짜_시간.csv` 이름으로 저장              |

---

### 📊 성능

* KoELECTRA-small 기준 AUC: `0.79095`
* 전처리 없이도 안정적 성능

---

### 🛠 참고 사항

* 실행 전, `train.csv`, `test.csv`, `sample_submission.csv` 파일 경로를 꼭 확인하세요.
* 모델은 소형 버전을 사용하므로 \*\*Colab 없이도 로컬(Mac mini)\*\*에서 동작합니다.
* 점수 향상을 원할 경우 `koelectra-base` 또는 `large` 모델로 교체 가능합니다.

---

### 🙋‍♂️ 개선 아이디어

* Stratified K-Fold 적용
* Label Smoothing Loss 도입
* Ensemble with TF-IDF + XGBoost

---

필요하다면 base 모델용 README도 함께 제공해드릴 수 있어요. 계속 개선해 나가고 싶으시면 말씀해주세요!
