---
Title: "Visual Question Answering with Multimodal Architectures"
Description: "VQA system using answerability classification and answer generation models on the VizWiz dataset"
---

# Visual Question Answering with Multimodal Architectures

This project presents a **Visual Question Answering (VQA)** system using two distinct multimodal deep learning models. It uses both images and natural language questions to either:
- Predict whether a question is answerable (Binary classification)
- Generate the best answer from 301 predefined classes (Multiclass classification)

Built and evaluated on the **real-world VizWiz dataset**, this project demonstrates attention-based fusion and fine-tuned CNNs for robust image-question reasoning.

---

## Model Architectures

### 1. Answerability Classification Model

A binary classifier that predicts if a given image-question pair is answerable.

**Architecture:**

- **Image Pipeline**
  - CNN with 4 Conv2d blocks + MaxPool, BatchNorm, ReLU
  - Flatten → Linear layer (32768 → hidden_dim)
- **Question Pipeline**
  - GloVe vector (50d) → Linear → ReLU + Dropout
- **Cross Attention**
  - Multi-headed attention (Image as Query, Question as Key/Value)
- **Fusion & Classification**
  - FC → ReLU + Dropout → FC → Sigmoid

**Input**: Image tensor `(3×128×128)`, GloVe question embedding `(50-d)`  
**Output**: Binary logit → Sigmoid → [0, 1]

---

### 2. Answer Generation Model

Multiclass classifier that predicts one of the **301** answer classes.

**Architecture:**

- **Image Pipeline**
  - Fine-tuned **ResNet18** (last 4 layers only)
  - FC → ReLU + Dropout (x2)
- **Question Pipeline**
  - GloVe vector → FC → ReLU + Dropout
- **Cross Attention**
  - Multi-head attention (Image as Query, Question as Key/Value)
- **Fusion & Classification**
  - Concatenate attended features → FC → LayerNorm → ReLU + Dropout (x2)
  - FC → 301 class logits

**Input**: Image tensor `(3×128×128)`, GloVe vector  
**Output**: 301 class scores (softmax during evaluation)

---

## Model Performance

| Model                    | Accuracy |
|--------------------------|----------|
| Answerability Classifier | **60.0%** |
| Answer Generator         | **55.3%** |


No pre-trained language models (e.g., BERT, GPT) were used, demonstrating the effectiveness of lightweight, interpretable multimodal fusion techniques in VQA settings.

---

## Hyperparameter Tuning (Optuna)

| Model                    | Best Params |
|--------------------------|-------------|
| Answerability Classifier | Adam, LR: `0.0048`, Hidden Dim: `512`, Batch: `32` |
| Answer Generator         | Adam, LR: `0.00249`, Hidden Dim: `512`, Batch: `16`, Heads: `4` |

**Trends:**
- ResNet18 transfer learning boosted accuracy
- More attention heads improved multimodal representation
- Smaller batch sizes improved generalization
- Adam outperformed SGD and AdamW

---

## Technologies Used

- Python, PyTorch
- ResNet18 (torchvision)
- GloVe Embeddings (glove.6B.50d)
- spaCy (text preprocessing)
- Optuna (hyperparameter optimization)
- NumPy, Matplotlib, Pandas

---

## Author

**ManiDatta**  
Master’s in Data Science @ University of Colorado Boulder  
[GitHub](https://github.com/Manidatta1)


**Full Report**

For a detailed explanation of the experiment design, model architecture, hyperparameter tuning, result interpretation, and additional insights, please refer to document Visual Question Answering.docx the included in this repository.


