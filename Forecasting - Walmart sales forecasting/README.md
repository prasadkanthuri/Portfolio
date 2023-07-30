# Forecasting  - Walmart Sales

### Introduction
This project aims to develop a Machine Learning (ML) model for Walmart's sales forecasting. The model will be trained on historical sales data and will predict future sales, enabling better inventory management and planning for the retail giant.

### Project Overview
Sales forecasting is crucial for businesses like Walmart to optimize their inventory, anticipate demand, and improve overall efficiency. By leveraging ML techniques, this project seeks to create a reliable and accurate sales forecasting model for Walmart's various stores and products.

### Dataset
The dataset used for this project consists of historical sales data from various Walmart stores. Each data point includes information such as store number, date, Sales on the given day, Temperature, CPI, Fuel price and Unemployment rate . 

### Data Preprocessing
Before feeding the data into the ML model, several preprocessing steps have been applied to ensure the data is in an appropriate format for training.

### Exploratory Data Analysis (EDA)
EDA has been performed on the dataset to gain insights into the sales patterns, seasonality, and correlations between different features. Visualizations and statistical analysis were used to better understand the data, which helped in making informed decisions during model development.

### Model Architecture
For this sales forecast model, we have chosen a FBprophet model. Because of Limited availability of the data points, LSTM model is not implemented.

#### Training
The model has been trained on a portion of the dataset using historical sales data. Other important variables that are considerably affecting the sales are included in training the model.

#### Evaluation
The model's performance was evaluated on a separate test dataset that was not used during training. Common evaluation metrics, such as Mean Absolute Percentage Error (MAPE) was calculated to assess the model's accuracy and effectiveness.

![fbprophet](https://github.com/prasadkanthuri/Portfolio/assets/135444495/1ff91b8b-a1ab-41d0-8664-ec95f05cf61c)
![actual vs forecast](https://github.com/prasadkanthuri/Portfolio/assets/135444495/75eb42ba-b288-4d76-8671-3b6eedc4ae86)
