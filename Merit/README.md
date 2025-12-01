# Merit: Multi-agent LLM-enhanced Time Series Representation Learning

시계열 데이터의 표현 학습을 위한 다중 에이전트 LLM 기반 프레임워크

## Overview

Merit는 LLM(Large Language Model) 에이전트를 활용하여 시계열 데이터의 고품질 표현(representation)을 학습하는 프레임워크입니다. Retrieval, Augmentation, Review 에이전트를 통해 contrastive learning을 수행합니다.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Merit Framework                         │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Retrieval   │  │ Augmentation │  │    Review    │       │
│  │    Agent     │──│    Agent     │──│    Agent     │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│         │                 │                 │               │
│         └─────────────────┼─────────────────┘               │
│                           ▼                                 │
│                   ┌──────────────┐                          │
│                   │ TCN Encoder  │                          │
│                   └──────────────┘                          │
│                           │                                 │
│                           ▼                                 │
│                   ┌──────────────┐                          │
│                   │ Contrastive  │                          │
│                   │   Learning   │                          │
│                   └──────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
Merit/
├── src/
│   ├── train.py          # 메인 학습 스크립트
│   ├── encoder.py        # TCN 인코더
│   ├── agents.py         # LLM 에이전트 (Retrieval, Augmentation, Review)
│   ├── data_loader.py    # 데이터 로더
│   ├── utils.py          # 유틸리티 함수
│   └── downstream/       # 다운스트림 태스크
│       ├── heads.py      # Classification, Clustering, Anomaly Detection, etc.
│       ├── evaluate.py   # 평가 로직
│       └── visualize.py  # 시각화
├── saved_models/         # 학습된 인코더 모델
├── saved_reps/           # 학습된 representations
├── downstream_results/   # 다운스트림 평가 결과
├── logs/                 # 학습 로그
├── run_all_training.sh   # 전체 학습 스크립트
└── evaluate_downstream.py # 다운스트림 평가 스크립트
```

## Supported Datasets

| Dataset | Samples | Channels | Length | Tasks |
|---------|---------|----------|--------|-------|
| AtrialFibrillation | 30 | 2 | 640 | Classification, Anomaly Detection, Forecasting |
| StandWalkJump | 27 | 4 | 2500 | Classification, Clustering, Anomaly Detection |
| ArticularyWordRecognition | 575 | 9 | 144 | Classification |
| NATOPS | 360 | 24 | 51 | Classification |
| PenDigits | 10992 | 2 | 8 | Classification, Clustering |
| UWaveGestureLibrary | 440 | 3 | 315 | Classification |
| PEMS-SF | 440 | 963 | 144 | Anomaly Detection, Forecasting, Imputation |

## Installation

```bash
# 가상환경 생성
conda create -n MERIT python=3.10
conda activate MERIT

# 의존성 설치
pip install -r requirements.txt
```

## Usage

### Training

```bash
# 단일 데이터셋 학습
python -m src.train \
    --dataset_name Epilepsy \
    --root_path /path/to/dataset \
    --llm_path /path/to/llama-3.1-8b-instruct \
    --gpu 0 \
    --epochs 10

# 전체 데이터셋 학습 (GPU 병렬)
./run_all_training.sh
```

### Downstream Evaluation

```bash
python evaluate_downstream.py \
    --reps_dir ./saved_reps \
    --output_dir ./downstream_results
```

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset_name` | BasicMotions | 데이터셋 이름 |
| `--root_path` | ./datasets/ | 데이터셋 루트 경로 |
| `--llm_path` | - | LLM 모델 경로 |
| `--gpu` | 0 | GPU 디바이스 ID |
| `--epochs` | 100 | 학습 에폭 수 |
| `--lr` | 1e-3 | 학습률 |
| `--weight_decay` | 5e-4 | Weight decay |
| `--bank_size` | 100 | Memory bank 크기 |
| `--repr_dim` | 320 | Representation 차원 |

## Downstream Tasks

### 1. Classification
- SVM, KNN, Random Forest
- Metrics: Accuracy, F1 Score, Precision, Recall

### 2. Clustering
- K-Means
- Metrics: NMI, RI, ARI, Silhouette Score

### 3. Anomaly Detection
- Isolation Forest, One-Class SVM
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC

### 4. Forecasting
- Ridge Regression
- Metrics: MSE, RMSE, MAE, R²

### 5. Imputation
- KNN-based Imputation
- Metrics: MSE, RMSE, MAE

## Results

자세한 결과는 `downstream_results/downstream_results.md` 참조

### Classification Summary
| Dataset | Accuracy |
|---------|----------|
| PenDigits | 0.5121 |
| StandWalkJump | 0.5000 |
| UWaveGestureLibrary | 0.2727 |
| NATOPS | 0.2639 |
| ArticularyWordRecognition | 0.2174 |
| AtrialFibrillation | 0.1667 |

## License

MIT License

## References

- Merit: Multi-agent LLM-enhanced Time Series Representation Learning
