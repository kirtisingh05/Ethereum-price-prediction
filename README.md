#  Ethereum Price Prediction Web App

A full-stack, AI-powered web application that forecasts Ethereum (ETH) prices using a Long Short-Term Memory (LSTM) neural network. The application features a Flask backend for handling predictions and a React frontend for visualizing real-time price forecasts.

 # Project Overview

This project demonstrates how deep learning can be applied to financial time series forecasting. It pulls historical Ethereum data from Yahoo Finance, processes it, and uses a pre-trained LSTM model to predict future prices. The results are then presented in a clean and responsive React interface**.

# Features

- Real-time Prediction of Ethereum prices
- Interactive Charts showing historical and forecasted data
- LSTM Neural Network built with TensorFlow/Keras
- REST API built with Flask for model inference
- React Frontend to fetch and display prediction results
- Modular Codebase: Easy to extend for other assets like BTC or stocks

# Tech Stack

| Layer       | Technologies                          |
|-------------|----------------------------------------|
| Frontend    | React, Axios, TailwindCSS, Recharts    |
| Backend     | Flask, Python, yfinance, NumPy, Pandas |
| Model       | TensorFlow, Keras, LSTM                |
| Deployment  | (Optional) Docker, Render, Vercel      |
