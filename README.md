# UTKFace-Age-and-Gender-Prediction-with-Deep-Learning

## Overview
This project utilizes the UTKFace dataset to analyze and predict facial attributes such as age, gender, and ethnicity using deep learning techniques. The dataset contains over 20,000 images with annotations covering a wide age range (0-116 years), making it suitable for tasks like age estimation, gender classification, and facial recognition.

## Features
- **Age Estimation**: Predicts the age of individuals using convolutional neural networks (CNNs).
- **Gender Classification**: Determines the gender of individuals based on facial features.
- **Ethnicity Classification**: Classifies individuals into different ethnic groups.
- **Facial Landmark Localization**: Detects key facial landmarks for further analysis.
- **Data Augmentation**: Enhances training data to improve model generalization.

## Dataset
- **Name**: UTKFace
- **Size**: 20,000+ images
- **Annotations**: Age, gender, ethnicity
- **Variations**: Different poses, expressions, lighting conditions, and occlusions
- **Source**: [UTKFace Dataset](https://susanqq.github.io/UTKFace/)

## Technologies Used
- **Programming Language**: Python
- **Deep Learning Frameworks**: TensorFlow, Keras, PyTorch
- **Libraries**: OpenCV, NumPy, Pandas, Matplotlib, Scikit-Learn

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/k-adi007/UTKFace-Age-and-Gender-Prediction-with-Deep-Learning.git
   cd UTKFace-Age-and-Gender-Prediction-with-Deep-Learning
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Download the UTKFace dataset and place it in the `data/` directory.

## Usage
### Training the Model
Run the following command to train the model:
```sh
python train.py --epochs 50 --batch_size 32 --learning_rate 0.001
```

### Evaluating the Model
To test the model on unseen data:
```sh
python evaluate.py --model checkpoint/model.pth --dataset test
```

### Running Inference
Use the trained model for age and gender prediction:
```sh
python predict.py --image sample.jpg
```

## Results
- Achieved **6.5 MEA** in age estimation.
- Gender classification reached **90% accuracy**.

## Future Work
- Implement real-time face analysis using OpenCV.
- Improve model robustness with additional datasets.
- Explore transfer learning techniques for better generalization.

## Acknowledgments
- UTKFace dataset creators
- TensorFlow, PyTorch, and OpenCV communities
