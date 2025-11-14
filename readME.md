
# Deep Learning Lab Experiments

This repository contains implementations of various deep learning architectures using TensorFlow/Keras for different tasks including text classification, image classification, regression, and time series prediction.

## 📋 Table of Contents
- [Overview](#overview)
- [Experiments](#experiments)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)

## 🔍 Overview

This project demonstrates the implementation and comparison of fundamental deep learning architectures:
- **LSTM** for text classification
- **MLP** for regression tasks
- **RNN** for sentiment analysis
- **CNN** for image classification

## 🧪 Experiments

### 1. LSTM - SMS Spam Classification
- **Architecture**: LSTM with Embedding layer
- **Dataset**: SMS Spam Collection
- **Task**: Binary classification (spam/ham detection)
- **Key Features**:
  - Text preprocessing and tokenization
  - Sequence padding
  - Embedding layer for word representation
  - LSTM for sequential pattern learning

### 2. Multilayer Perceptron (MLP) - Sleep Health Prediction
- **Architecture**: Feedforward Neural Network
- **Dataset**: Sleep Health and Lifestyle Dataset
- **Task**: Regression/Classification
- **Key Features**:
  - Dense layers with ReLU activation
  - StandardScaler normalization
  - Dropout for regularization
  - Adam optimizer

### 3. RNN - IMDB Sentiment Analysis
- **Architecture**: Simple RNN with Embedding
- **Dataset**: IMDB Movie Reviews (50k reviews)
- **Task**: Binary sentiment classification (positive/negative)
- **Key Features**:
  - Top 20,000 vocabulary
  - Sequence padding (maxlen=200)
  - Embedding dimension: 64
  - SimpleRNN layer with 64 units

### 4. CNN - CIFAR-10 Image Classification
- **Architecture**: 3-block CNN with Batch Normalization
- **Dataset**: CIFAR-10 (10 classes, 60k images)
- **Task**: Multi-class image classification
- **Model Summary**:
```
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃ Param #       ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 32, 32, 32)     │ 896           │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization             │ (None, 32, 32, 32)     │ 128           │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 16, 16, 32)     │ 0             │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 16, 16, 32)     │ 0             │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 16, 16, 64)     │ 18,496        │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_1           │ (None, 16, 16, 64)     │ 256           │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 8, 8, 64)       │ 0             │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (Dropout)             │ (None, 8, 8, 64)       │ 0             │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_2 (Conv2D)               │ (None, 8, 8, 128)      │ 73,856        │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_2           │ (None, 8, 8, 128)      │ 512           │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_2 (MaxPooling2D)  │ (None, 4, 4, 128)      │ 0             │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_2 (Dropout)             │ (None, 4, 4, 128)      │ 0             │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 2048)           │ 0             │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 512)            │ 1,049,088     │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_3 (Dropout)             │ (None, 512)            │ 0             │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 10)             │ 5,130         │
└─────────────────────────────────┴────────────────────────┴───────────────┘

Total params: 1,148,362 (4.38 MB)
Trainable params: 1,147,914 (4.38 MB)
Non-trainable params: 448 (1.75 KB)
```

**Key Features**:
- 3 Convolutional blocks (32 → 64 → 128 filters)
- Batch Normalization after each Conv2D
- MaxPooling2D for spatial downsampling
- Dropout (0.25 in conv blocks, 0.5 in dense)
- Dense layer with 512 units
- Confusion matrix visualization

## 🛠️ Requirements
```
tensorflow>=2.10.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
pandas>=1.3.0
statsmodels>=0.13.0
```

## 🚀 Usage

Each experiment is contained in a separate notebook/script:
```bash
# Run LSTM SMS Classification
python lstm_sms.py

# Run MLP Sleep Health Prediction
python mlp_sleep.py

# Run RNN IMDB Sentiment Analysis
python rnn_imdb.py

# Run CNN CIFAR-10 Classification
python cnn_cifar10.py
```

## 📊 Results

| Model | Dataset | Task | Metric | Score |
|-------|---------|------|--------|-------|
| LSTM | SMS | Classification | Accuracy | ~98% |
| MLP | Sleep Health | Regression | MAE | Variable |
| RNN | IMDB | Sentiment | Accuracy | ~85% |
| CNN | CIFAR-10 | Image Classification | Accuracy | ~75-80% |

## 📁 Project Structure
```
deep-learning-experiments/
│
├── lstm_sms.py              # LSTM for SMS classification
├── mlp_sleep.py             # MLP for sleep health prediction
├── rnn_imdb.py              # RNN for sentiment analysis
├── cnn_cifar10.py           # CNN for image classification
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🎯 Key Takeaways

- **LSTM** excels at sequential text data with temporal dependencies
- **MLP** provides a simple baseline for structured tabular data
- **RNN** captures sequential patterns in text for sentiment analysis
- **CNN** with BatchNorm and Dropout prevents overfitting on images

## 📝 License

MIT License

## 👤 Author

Your Name - [GitHub Profile](https://github.com/yourusername)

---

*This project was created as part of a Deep Learning lab course.*
