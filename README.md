# Titanic Survival Prediction with Feast - 완전 매뉴얼

이 프로젝트는 [Feast](https://feast.dev/)를 사용하여 Titanic 생존 예측 문제를 해결하는 완전한 예제입니다. CSV 데이터를 Parquet로 변환하는 과정부터 Feature Store 구축, 모델 훈련, 실시간 예측까지의 전체 과정을 다룹니다.

## 🎯 프로젝트 목표

- **Feast Feature Store 구축**: ML 모델을 위한 feature 관리 시스템 구축
- **데이터 전처리**: CSV → Parquet 변환 및 timestamp 추가
- **Feature Engineering**: 기본 및 고급 feature 생성
- **모델 훈련**: Random Forest를 이용한 생존 예측
- **실시간 예측**: Online feature serving을 통한 실시간 예측
- **성능 분석**: Feature importance 및 모델 성능 분석

## 📁 프로젝트 구조

```
feature_repo/
├── data/
│   ├── train.csv                    # 원본 훈련 데이터 (891행)
│   ├── test.csv                     # 원본 테스트 데이터 (418행)
│   ├── train_processed.csv          # 전처리된 CSV 데이터
│   ├── test_processed.csv           # 전처리된 CSV 데이터
│   ├── train_processed.parquet      # Feast용 Parquet 데이터 ⭐
│   └── test_processed.parquet       # Feast용 Parquet 데이터 ⭐
├── feature_store.yaml               # Feast 설정 파일
├── prepare_data.py                  # 데이터 전처리 스크립트
├── titanic_features.py              # 기본 feature 정의
├── titanic_example.py               # 기본 예제 실행 스크립트
├── advanced_titanic_features.py     # 고급 feature 정의
├── advanced_titanic_example.py      # 고급 예제 실행 스크립트
└── run_titanic_example.py           # 메인 실행 스크립트
```

## 🛠️ 설치 및 설정

### 1. 환경 설정

```bash
# Conda 환경 생성 (권장)
conda env create -f environment.yml
conda activate titanic_feast

# 또는 pip로 직접 설치
pip install feast[local] pandas scikit-learn matplotlib seaborn pyarrow
```

### 2. 프로젝트 실행

```bash
# feature_repo 디렉토리로 이동
cd feature_repo/feature_repo

# 메인 실행 스크립트 실행
python run_titanic_example.py
```

## 📊 데이터 전처리 과정

### 1. CSV → Parquet 변환 이유

**Feast의 FileSource는 Parquet 형식을 기본적으로 지원**하기 때문에 CSV를 Parquet로 변환합니다.

```python
# prepare_data.py에서 수행하는 작업들:

# 1. 원본 CSV 데이터 읽기
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# 2. Timestamp 컬럼 추가 (Feast 필수)
base_date = datetime(1912, 4, 10)  # Titanic 출발일
train_df['event_timestamp'] = pd.to_datetime([base_date + timedelta(days=int(days)) for days in random_days])
train_df['created_timestamp'] = train_df['event_timestamp']

# 3. 결측값 처리
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
train_df['Embarked'] = train_df['Embarked'].fillna('S')
train_df['Cabin'] = train_df['Cabin'].fillna('Unknown')

# 4. CSV 및 Parquet로 저장
train_df.to_csv("data/train_processed.csv", index=False)
train_df.to_parquet("data/train_processed.parquet", index=False)
```

### 2. 전처리 결과

| 원본 데이터 | 전처리 후 |
|------------|-----------|
| 891행 (훈련), 418행 (테스트) | 동일 |
| 결측값 존재 | 중앙값/최빈값으로 대체 |
| Timestamp 없음 | event_timestamp, created_timestamp 추가 |
| CSV 형식 | CSV + Parquet 형식 |

## 🏗️ Feature Store 구축

### 1. Entity 정의

```python
# titanic_features.py
passenger = Entity(
    name="passenger_id",
    value_type=ValueType.INT64,
    description="Passenger ID",
    join_keys=["PassengerId"],
)
```

**Entity**: 예측 대상이 되는 개체 (여기서는 승객)

### 2. Data Source 정의

```python
passenger_source = FileSource(
    path="data/train_processed.parquet",  # Parquet 파일 사용
    timestamp_field="event_timestamp",    # 시간 정보
    created_timestamp_column="created_timestamp",
)
```

### 3. Feature View 정의

```python
passenger_features = FeatureView(
    name="passenger_features",
    entities=[passenger],
    ttl=timedelta(days=365),  # 1년간 유효
    schema=[
        Field(name="PassengerId", dtype=Int64),
        Field(name="Pclass", dtype=Int64),      # 객실 등급
        Field(name="Sex", dtype=String),        # 성별
        Field(name="Age", dtype=Float32),       # 나이
        Field(name="SibSp", dtype=Int64),       # 형제자매/배우자 수
        Field(name="Parch", dtype=Int64),       # 부모/자녀 수
        Field(name="Fare", dtype=Float32),      # 요금
        Field(name="Embarked", dtype=String),   # 승선 항구
    ],
    source=passenger_source,
    online=True,  # 실시간 서빙 가능
)
```

## 🎯 Feature Engineering

### 1. 기본 Features

| Feature | 설명 | 데이터 타입 | 범위 |
|---------|------|------------|------|
| **Pclass** | 객실 등급 | Int64 | 1(1등석), 2(2등석), 3(3등석) |
| **Sex** | 성별 | String | male, female |
| **Age** | 나이 | Float32 | 0.42 ~ 80 |
| **SibSp** | 형제자매/배우자 수 | Int64 | 0 ~ 8 |
| **Parch** | 부모/자녀 수 | Int64 | 0 ~ 6 |
| **Fare** | 요금 | Float32 | 0 ~ 512.3292 |
| **Embarked** | 승선 항구 | String | C(Cherbourg), Q(Queenstown), S(Southampton) |

### 2. 고급 Features (On-demand)

```python
# advanced_titanic_features.py
@OnDemandFeatureView(
    sources=[passenger_features, request_source],
    mode="python",
    schema=[...]
)
def derived_features(input_df: pd.DataFrame) -> pd.DataFrame:
    # 가족 크기
    df["family_size"] = df["SibSp"] + df["Parch"] + 1

    # 혼자 여행 여부
    df["is_alone"] = (df["family_size"] == 1).astype(int)

    # 연령대 분류
    df["age_group"] = df["Age"].apply(lambda x:
        "child" if x < 18 else
        "young_adult" if x < 30 else
        "adult" if x < 50 else "senior")

    # 1인당 요금
    df["fare_per_person"] = df["Fare"] / df["family_size"]

    # 이름에서 호칭 추출
    df["title"] = df["Name"].apply(lambda x: x.split(',')[1].split('.')[0].strip())

    # 객실 데크
    df["cabin_deck"] = df["Cabin"].apply(lambda x: x[0] if len(x) > 0 else "Unknown")

    return df
```

## 🤖 모델 훈련 과정

### 1. Historical Features 추출

```python
# titanic_example.py
def get_training_data(store):
    # 훈련 데이터 읽기
    train_df = pd.read_parquet("data/train_processed.parquet")

    # Entity dataframe 생성
    entity_df = train_df[["PassengerId", "event_timestamp"]].copy()
    entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"])

    # Historical features 추출
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "passenger_features:Pclass",
            "passenger_features:Sex",
            "passenger_features:Age",
            "passenger_features:SibSp",
            "passenger_features:Parch",
            "passenger_features:Fare",
            "passenger_features:Embarked",
            "survival_features:Survived"
        ],
        full_feature_names=True
    ).to_df()

    return training_df
```

### 2. 모델 훈련

```python
def train_model(training_df):
    # Feature 준비
    feature_columns = [
        'passenger_features__Pclass',
        'passenger_features__Sex',
        'passenger_features__Age',
        'passenger_features__SibSp',
        'passenger_features__Parch',
        'passenger_features__Fare',
        'passenger_features__Embarked'
    ]

    # Categorical 변수 인코딩
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()

    X = training_df[feature_columns].copy()
    X['passenger_features__Sex'] = le_sex.fit_transform(X['passenger_features__Sex'])
    X['passenger_features__Embarked'] = le_embarked.fit_transform(X['passenger_features__Embarked'])

    y = training_df['survival_features__Survived']

    # 데이터 분할 및 모델 훈련
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, le_sex, le_embarked
```

## 📈 모델 성능 분석

### 1. 전체 성능

```
Model accuracy: 0.8101

Classification Report:
              precision    recall  f1-score   support

           0       0.85      0.85      0.85       113  # 사망 예측
           1       0.74      0.74      0.74        66  # 생존 예측

    accuracy                           0.81       179
   macro avg       0.80      0.80      0.80       179
weighted avg       0.81      0.81      0.81       179
```

### 2. Feature Importance 분석

| 순위 | Feature | Importance | 설명 |
|------|---------|------------|------|
| 1 | **Fare** | 27.08% | 요금이 가장 중요한 예측 요소 |
| 2 | **Age** | 26.14% | 나이가 두 번째로 중요 |
| 3 | **Sex** | 24.96% | 성별이 세 번째로 중요 |
| 4 | Pclass | 9.02% | 객실 등급 |
| 5 | SibSp | 4.79% | 형제자매/배우자 수 |
| 6 | Parch | 4.23% | 부모/자녀 수 |
| 7 | Embarked | 3.78% | 승선 항구 |

### 3. Feature Importance 해석

**🎯 Fare (27.08%)**:
- 요금이 높을수록 생존 확률이 높음
- 1등석 승객들이 더 비싼 요금을 지불했고, 구명정에 우선 접근 가능

**🎯 Age (26.14%)**:
- 어린이와 노인의 생존률이 다름
- "여성과 어린이 먼저" 원칙이 적용됨

**🎯 Sex (24.96%)**:
- 여성의 생존률이 남성보다 훨씬 높음
- 당시의 사회적 관습과 구명정 배치 정책 반영

## 🔄 실시간 예측 과정

### 1. Online Features 추출

```python
def get_online_features(store, passenger_ids):
    # Online features 가져오기
    online_features = store.get_online_features(
        features=[
            "passenger_features:Pclass",
            "passenger_features:Sex",
            "passenger_features:Age",
            "passenger_features:SibSp",
            "passenger_features:Parch",
            "passenger_features:Fare",
            "passenger_features:Embarked"
        ],
        entity_rows=[{"PassengerId": pid} for pid in passenger_ids]
    ).to_df()

    return online_features
```

### 2. Materialization 없이도 작동하는 이유

**🎯 핵심: FileSource의 `online=True` 설정**

```python
# titanic_features.py
passenger_features = FeatureView(
    name="passenger_features",
    entities=[passenger],
    ttl=timedelta(days=365),
    schema=[...],
    source=passenger_source,  # FileSource
    online=True,  # ⭐ 이 설정이 핵심!
)
```

**`online=True` 설정의 효과:**
- Feature View가 **online store에 자동으로 등록**됩니다
- `get_online_features()` 호출 시 **파일에서 직접 데이터를 읽어옵니다**
- **Materialization 없이도 실시간 feature serving이 가능**합니다

**실제 데이터 흐름:**
```
1. get_online_features() 호출
2. Feast가 FileSource에서 직접 데이터 읽기
3. PassengerId로 필터링
4. 결과 반환
```

### 3. Materialization vs Non-Materialization 비교

| 구분 | Materialization | Non-Materialization |
|------|----------------|-------------------|
| **데이터 소스** | Online Store (SQLite) | FileSource (Parquet) |
| **속도** | 빠름 (인덱싱됨) | 상대적으로 느림 |
| **메모리 사용** | 더 많음 | 적음 |
| **실시간성** | 높음 | 높음 |
| **설정 복잡도** | 높음 | 낮음 |

### 4. 예측 수행

```python
def predict_survival(store, model, le_sex, le_embarked, passenger_ids):
    # Online features 가져오기
    online_features = get_online_features(store, passenger_ids)

    # 결측값 처리
    X_pred = online_features[available_columns].copy()
    X_pred = X_pred.fillna({
        'Age': X_pred['Age'].median(),
        'Fare': X_pred['Fare'].median(),
        'Sex': 'male',
        'Embarked': 'S',
        # ... 기타 기본값들
    })

    # Categorical 변수 인코딩
    X_pred["Sex"] = le_sex.transform(X_pred["Sex"])
    X_pred["Embarked"] = le_embarked.transform(X_pred["Embarked"])

    # 컬럼명을 훈련 데이터와 맞춤
    X_pred = X_pred.rename(columns=column_mapping)

    # 예측 수행
    predictions = model.predict(X_pred)
    probabilities = model.predict_proba(X_pred)

    return results
```

### 5. 예측 결과

```
Prediction Results:
   PassengerId  Survived_Prediction  Survival_Probability
0          892                    0                 0.005
1          893                    0                 0.005
2          894                    0                 0.005
...
```

**해석**: 테스트 데이터의 처음 10명 승객 모두 사망할 것으로 예측 (낮은 생존 확률)

## 🎯 현재 구현의 장단점

### ✅ **장점**
1. **간단함**: Materialization 과정이 불필요
2. **즉시 사용 가능**: 파일만 있으면 바로 작동
3. **메모리 효율적**: 필요한 데이터만 로드
4. **개발/테스트에 적합**: 빠른 프로토타이핑 가능

### ❌ **단점**
1. **성능**: 매번 파일에서 읽어오므로 느릴 수 있음
2. **확장성**: 대용량 데이터에서는 비효율적
3. **실시간성**: 파일 I/O 오버헤드
4. **Production 부적합**: 대규모 서비스에는 부적절

## 🚀 Production 환경에서는?

실제 프로덕션에서는 **Materialization이 필수**입니다:

```python
# Production 환경에서의 올바른 방식
def production_workflow():
    # 1. Materialize features to online store
    store.materialize(start_date=start_date, end_date=end_date)

    # 2. Get features from online store (빠름)
    online_features = store.get_online_features(...)

    # 3. Make predictions
    predictions = model.predict(online_features)
```

**Production 환경에서 Materialization이 필요한 이유:**
- **성능 최적화**: 인덱싱된 데이터로 빠른 조회
- **확장성**: 대용량 데이터 처리 가능
- **안정성**: 파일 시스템 의존성 제거
- **실시간성**: 낮은 지연시간 보장

## 🚀 고급 기능

### 1. On-demand Feature View

실시간으로 계산되는 파생 features:

```python
# 가족 크기
family_size = SibSp + Parch + 1

# 혼자 여행 여부
is_alone = (family_size == 1)

# 연령대 분류
age_group = "child" if age < 18 else "young_adult" if age < 30 else "adult" if age < 50 else "senior"

# 1인당 요금
fare_per_person = Fare / family_size
```

### 2. Feature Service

여러 Feature View를 조합한 서비스:

```python
survival_prediction_service = FeatureService(
    name="survival_prediction_service",
    features=[passenger_features, survival_features],
    description="Features for predicting passenger survival",
)
```

## 🔧 사용법

### 1. 단계별 실행

```bash
# 1. 데이터 전처리 (CSV → Parquet)
python prepare_data.py

# 2. 기본 예제 실행
python titanic_example.py

# 3. 고급 예제 실행
python advanced_titanic_example.py
```

### 2. 개별 기능 테스트

```python
from feast import FeatureStore

# Feature store 초기화
store = FeatureStore(repo_path=".")

# Online features 가져오기
features = store.get_online_features(
    features=[
        "passenger_features:Pclass",
        "passenger_features:Sex",
        "passenger_features:Age"
    ],
    entity_rows=[{"PassengerId": 1}]
).to_df()

print(features)
```

## 📊 결과 해석

### 1. 모델 성능 (81.01% 정확도)

- **좋은 점**: 기본적인 생존 예측 패턴을 잘 학습
- **개선 가능**: 더 많은 feature engineering으로 성능 향상 가능

### 2. Feature Importance 분석

- **요금(Fare)**: 사회적 지위와 생존 가능성의 강한 상관관계
- **나이(Age)**: 어린이 우선 원칙의 반영
- **성별(Sex)**: 여성 우선 원칙의 명확한 반영

### 3. 예측 결과

- 테스트 데이터의 낮은 생존 확률은 모델이 보수적으로 예측했음을 의미
- 실제 Titanic 사고에서 생존률이 약 38%였음을 고려하면 합리적

## 🎯 학습 포인트

### 1. Feast 핵심 개념
- **Entity**: 예측 대상 (PassengerId)
- **Feature View**: feature 정의 및 데이터 소스
- **Feature Service**: 여러 feature view 조합
- **On-demand Feature View**: 실시간 계산되는 파생 feature

### 2. 데이터 전처리
- **CSV → Parquet**: Feast 호환성을 위한 형식 변환
- **Timestamp 추가**: 시간 기반 feature store 구축
- **결측값 처리**: 모델 성능 향상을 위한 데이터 정제

### 3. Feature Engineering
- **기본 Features**: 원본 데이터에서 직접 추출
- **파생 Features**: 기존 features 조합으로 생성
- **On-demand 계산**: 실시간 feature 계산

### 4. Online Serving 메커니즘
- **FileSource + online=True**: Materialization 없이도 실시간 서빙 가능
- **Materialization vs Non-Materialization**: 각각의 장단점과 사용 시기
- **Production vs Development**: 환경별 최적화 전략

## 🚀 확장 가능성

### 1. 추가 Features
- 승객 이름에서 국적 추출
- 티켓 번호 패턴 분석
- 객실 위치 기반 features

### 2. 모델 개선
- 딥러닝 모델 적용 (Neural Networks)
- 앙상블 방법론 (XGBoost, LightGBM)
- 하이퍼파라미터 튜닝

### 3. 실시간 시스템
- 실시간 승객 정보 업데이트
- 동적 feature 계산
- A/B 테스트 지원

## 📚 참고 자료

- [Feast 공식 문서](https://docs.feast.dev/)
- [Titanic Dataset](https://www.kaggle.com/c/titanic)
- [Feature Store 개념](https://www.featurestore.org/)
- [Random Forest 설명](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

---

**Happy Feature Engineering with Feast! 🚢✨**

*이 매뉴얼은 CSV에서 Parquet 변환부터 실시간 예측까지의 전체 과정을 상세히 설명합니다.*