# Fashion MNIST Classifier

## Overview
This project implements a Convolutional Neural Network (CNN) model using TensorFlow/Keras to classify images from the Fashion MNIST dataset. The model leverages advanced techniques such as data augmentation, dropout layers, batch normalization, and L2 regularization to enhance performance and mitigate overfitting. Built using TensorFlow's **Functional API**, the model offers flexibility and scalability, making it easier to access intermediate layers for visualization and analysis.

---

## Dataset
- **Fashion MNIST**: A dataset consisting of 70,000 grayscale images of 10 categories of clothing items.
- **Image Size**: 28x28 pixels.
- **Classes**:
  1. T-shirt/top  
  2. Trouser  
  3. Pullover  
  4. Dress  
  5. Coat  
  6. Sandal  
  7. Shirt  
  8. Sneaker  
  9. Bag  
  10. Ankle boot  

---

## Features
1. **Data Augmentation**:
   - Utilized `ImageDataGenerator` for random transformations including rotation, shifting, shearing, and zooming.
   - Visualized augmented images to validate transformations.
   
2. **Model Architecture**:
   - Built using TensorFlow's **Functional API**.
   - Multiple convolutional layers with ReLU activation.
   - Incorporated **batch normalization** and **dropout layers** to improve generalization.
   - Applied **L2 regularization** to prevent overfitting.
   
3. **Callbacks**:
   - Early stopping to terminate training when validation loss stops improving.
   - Model checkpoint to save the best-performing model.
   - ReduceLROnPlateau for dynamic learning rate adjustment.
   
4. **Preprocessing**:
   - Normalized pixel values to [0, 1].
   - Reshaped data to include a channel dimension for compatibility with CNN.
   - Split data into training, validation, and testing sets with stratified sampling.
   
5. **Visualization**:
   - Plotted training and validation loss and accuracy.
   - Displayed sample predictions with true and predicted labels.
   - Generated a confusion matrix to evaluate per-class performance.
   - Showed misclassified images to understand model errors.
   - Printed a detailed classification report.
   - Visualized feature maps from intermediate convolutional layers.

---

## Running the Code
1. **Environment Setup**:
   - Ensure you have Python installed (preferably Python 3.6 or later).
   - Install necessary libraries using pip:
     ```bash
     pip install tensorflow matplotlib seaborn scikit-learn
     ```
   - Alternatively, use the provided `requirements.txt` (if available):
     ```bash
     pip install -r requirements.txt
     ```
   
2. **Using Google Colab**:
   - Open [Google Colab](https://colab.research.google.com/).
   - Upload the Jupyter Notebook file (`Fashion_MNIST_Classifier.ipynb`).
   - Run all cells sequentially to preprocess data, train the model, and evaluate performance.
   
3. **Local Execution**:
   - Clone the repository or download the notebook.
   - Navigate to the project directory.
   - Launch Jupyter Notebook:
     ```bash
     jupyter notebook
     ```
   - Open and run the `Fashion_MNIST_Classifier.ipynb` notebook.

---

## Results
### Expected Output:
Once the model is trained and evaluated, you should obtain results similar to the following classification report:

| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| T-shirt/top   | 0.89      | 0.77   | 0.83     | 1000    |
| Trouser       | 1.00      | 0.98   | 0.99     | 1000    |
| Pullover      | 0.90      | 0.75   | 0.82     | 1000    |
| Dress         | 0.89      | 0.89   | 0.89     | 1000    |
| Coat          | 0.76      | 0.85   | 0.80     | 1000    |
| Sandal        | 0.98      | 0.97   | 0.98     | 1000    |
| Shirt         | 0.62      | 0.74   | 0.67     | 1000    |
| Sneaker       | 0.91      | 0.98   | 0.94     | 1000    |
| Bag           | 0.98      | 0.98   | 0.98     | 1000    |
| Ankle boot    | 0.99      | 0.92   | 0.95     | 1000    |
| **Accuracy**  |           |        | **0.88** | **10000** |
| **Macro Avg** | 0.89      | 0.88   | 0.89     | 10000    |
| **Weighted Avg** | 0.89   | 0.88   | 0.89     | 10000    |

### Observations:
- The model performs exceptionally well on classes like **Trouser**, **Sandal**, **Bag**, and **Ankle boot** (F1-scores ~0.95-0.99).
- Performance is lower for **T-shirt/top** and **Shirt** (F1-scores ~0.67-0.83), indicating these classes could benefit from:
  1. **Additional data augmentation** focusing on rotations and flips to capture variations.
  2. **Class weighting** during training to balance misclassification.
  3. **Fine-tuning the learning rate** to improve model generalization.

---


