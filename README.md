# Fruit Classification Model

## Overview
This project focuses on training a deep learning model to classify fresh and rotten fruits using transfer learning, data augmentation, and fine-tuning techniques. The model is built upon a VGG16 base model pretrained on ImageNet and customized to fit the specific fruit classification task. It aims to achieve a validation accuracy of 92% or higher.

## Dataset
The dataset consists of six categories of fruits: fresh apples, fresh oranges, fresh bananas, rotten apples, rotten oranges, and rotten bananas. Each image is normalized and undergoes data augmentation using techniques such as rotation, shifting, shearing, zooming, and flipping.

## Model Architecture
The model begins with a VGG16 base model with pre-trained weights, followed by a Global Average Pooling layer and a densely connected layer with a softmax activation function. The output layer has six neurons to represent the fruit categories. The base model's layers are initially frozen to retain the pre-trained features, and then fine-tuned with a low learning rate.

## Training and Evaluation
The model is trained on the augmented dataset and evaluated on a separate validation dataset. The training process involves multiple epochs and the optimization of the loss function. Once the validation accuracy reaches 92% or higher, the model is evaluated using the `evaluate` function to ensure its accuracy.

## Usage
1. Clone this repository: `https://github.com/Amario1306619051/Fruit-ripeness-CNN`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Download and organize the dataset into the `data/fruits` directory.
4. Train the model: Run the Jupyter notebook or Python script provided in the repository.
5. Evaluate the model: Run the provided assessment script to assess the model's performance.

## Results
The trained model achieved a validation accuracy of [accuracy]% on the fruit classification task. The model's performance can be further improved by experimenting with different base models, hyperparameters, and augmentation techniques.

## Credits
The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/sriramr/fruits-fresh-and-rotten-for-classification).

---