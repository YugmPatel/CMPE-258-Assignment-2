# CMPE-258-Assignment-2

## Overview
This project includes three deep learning models implemented in Google Colab, each addressing different types of machine learning tasks:

1. **Classification Model:** A deep learning model for a classification task with evaluation metrics such as accuracy, precision, recall, and F1-score.
2. **Regression Model:** A deep learning model for predicting continuous values using regression analysis.
3. **Image Classification Model:** A convolutional neural network (CNN) designed for image classification.

Each model is thoroughly documented and includes all relevant artifacts, metrics, and performance analyses as demonstrated in class.

## Project Structure
- **Colab Notebooks:** Three separate Google Colab notebooks for classification, regression, and image classification.
- **Artifacts & Metrics:** Stored and logged using Weights & Biases (preferred) or TensorFlow TensorBoard.
- **Evaluation Metrics:**
  - Accuracy, Precision, Recall, and F1-score (for classification tasks)
  - ROC and Precision-Recall (PR) curves
  - Per-class examples and error analysis
  - Visualization of training/validation loss and accuracy curves
- **Video Walkthrough:** A detailed explanation of the models, metrics, and error analysis.
- **README.md:** This document explaining the project setup and execution.

## Requirements
The following dependencies are required to run the notebooks:
- Python 3.x
- TensorFlow
- Keras
- Weights & Biases (wandb) or TensorFlow TensorBoard
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

To install the required libraries, run:
```bash
pip install tensorflow keras wandb numpy pandas matplotlib scikit-learn
```

## Model Details

### 1. Classification Model
- Uses a deep neural network (DNN) to classify input data into discrete categories.
- Evaluates performance using accuracy, precision, recall, F1-score, and per-class metrics.
- Includes ROC and PR curves for performance visualization.
- Logs training artifacts using Weights & Biases or TensorBoard.

### 2. Regression Model
- Uses a deep learning model to predict continuous values.
- Evaluates performance using Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared score.
- Plots error distribution and loss curves.
- Logs artifacts for further analysis.

### 3. Image Classification Model
- Utilizes a convolutional neural network (CNN) for classifying images.
- Trained on a dataset such as CIFAR-10 or MNIST.
- Evaluates performance using accuracy, confusion matrix, and per-class analysis.
- Visualizes misclassified examples and per-class error analysis.

## Running the Notebooks
1. Open the respective Google Colab notebook.
2. Run all cells to train and evaluate the model.
3. Check Weights & Biases (wandb) or TensorBoard for real-time logging of metrics and artifacts.
4. Analyze the generated plots and reports for insights.

## Output Artifacts
- Trained model weights and logs.
- Performance metrics.
- Visualizations for training history, confusion matrix, and per-class error analysis.
- ROC and PR curves.
- Sample predictions with explanations.

## Video Walkthrough
A video walkthrough explaining the models, training process, evaluation metrics, and error analysis is included.

## Conclusion
This project provides an end-to-end deep learning pipeline for classification, regression, and image classification. The models are thoroughly evaluated with various metrics and visualizations, ensuring a comprehensive understanding of their performance.

