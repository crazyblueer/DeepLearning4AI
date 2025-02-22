This repository contains key machine learning and deep learning projects that I have done in the Deep Learning For AI class focusing on different tasks such as image processing, natural language processing (NLP), and time series forecasting. 
Below are the details of each project:

Projects

1. Image Processing: CNN-based ResNet-50 for Image Classification
   
Objective: This project focuses on building an image classification model using a Convolutional Neural Network (CNN) architecture based on ResNet-50.

Dataset: CIFAR-10 dataset, which consists of 60,000 images in 10 different classes.

Tools/Technologies:
TensorFlow for building and training the model.
Keras for easy-to-use neural network API.
Python libraries: NumPy, Matplotlib, etc.

Description: The model utilizes ResNet-50, a deep CNN with residual learning, to classify images into predefined categories from the CIFAR-10 dataset. The project also demonstrates training, evaluation, and performance optimization techniques.


3. Movie Review Classification: Text Feedback with RNN Models
   
Objective: This project implements models to classify text feedback from movie reviews, determining whether a review is positive or negative.

Approach: Used Recurrent Neural Networks (RNN) for sequence modeling with both Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) architectures.

Dataset: IMDB dataset containing 50,000 movie reviews labeled as positive or negative.

Tools/Technologies:
TensorFlow/Keras for deep learning models.
Python libraries: NLTK for text preprocessing, Matplotlib for visualizations.

Description: The project trains and compares LSTM and GRU models, implementing techniques like text tokenization, embedding layers, and dropout regularization to achieve high classification accuracy.


5. Stock Prediction: Time Series Forecasting with LSTM Models
   
Objective: The goal is to predict stock prices and forecast market trends using Long Short-Term Memory (LSTM) networks, a type of Recurrent Neural Network (RNN) designed for time series data.

Stock Markets: The project focuses on both the Nasdaq and the Vietnam stock market.

Tools/Technologies:
TensorFlow/Keras for building the LSTM model.

Python libraries: Pandas for data manipulation, NumPy for numerical computations, Matplotlib for plotting.
Yahoo Finance API for stock data collection.

Description: This project trains an LSTM model on historical stock price data to predict future stock prices. It also includes portfolio performance evaluation by backtesting the predicted prices.
