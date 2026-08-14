# 🏠 House Price Prediction

## 📌 Project Overview

This project is a Machine Learning regression project developed as part of the **CloudExify Data Science Month 2 Internship — Project 3**.

The goal is to predict the price of houses using property-related features such as:

* Area Size
* Number of Bedrooms
* Number of Bathrooms
* City
* Property Type

Two regression models are trained and compared:

1. Linear Regression
2. Random Forest Regression

The model with the better test R² score is selected as the final model for house-price prediction.

## 🎯 Project Objectives

The main objectives of this project are:

* Explore and understand the house-price dataset
* Handle missing and unrealistic data
* Convert property areas into a common unit (Marla)
* Encode categorical variables
* Split the dataset into training and testing sets
* Train a Linear Regression model
* Train a Random Forest Regression model
* Compare model performance
* Analyze feature importance
* Visualize house-price patterns
* Predict the price of a new house

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* VS Code

## 📂 Project Structure

```text
House-Price-Prediction/
│
├── house_price_prediction.py
├── house_prices.csv
├── README.md
```
## 📊 Dataset

The dataset contains information about residential properties.

Important features used in the model include:

| Feature         | Description                        |
| --------------- | ---------------------------------- |
| `Area_Marla`    | Property area converted to Marla   |
| `bedrooms`      | Number of bedrooms                 |
| `baths`         | Number of bathrooms                |
| `city`          | City where the property is located |
| `property_type` | Type of property                   |
| `price`         | House price in PKR                 |

Properties listed for sale are used for the prediction task.

### Area Conversion

The dataset contains different area units.

For consistency:

**1 Kanal = 20 Marla**

Therefore, Kanal values are converted into Marla before model training.

## 🧹 Data Preprocessing

The following preprocessing steps are performed:

1. Load the dataset using Pandas.
2. Check the dataset structure and missing values.
3. Remove rows with missing values in required columns.
4. Filter the dataset to include properties listed **For Sale**.
5. Convert property areas from Kanal to Marla.
6. Remove unrealistic price, area, bedroom, and bathroom values.
7. Select relevant features.
8. Convert categorical features into numerical values using one-hot encoding.
9. Split the data into training and testing sets.

The dataset is divided using:

```text
80% Training Data
20% Testing Data
```

## 🤖 Machine Learning Models

### 1. Linear Regression

Linear Regression is used as the baseline regression model.

It attempts to learn the relationship between the property features and house price.

### 2. Random Forest Regression

Random Forest Regression is used as a second model because it can capture more complex relationships between property characteristics and price.

The two models are evaluated using the test dataset.

## 📏 Model Evaluation

The following evaluation metrics are used:

### R² Score

R² measures how well the model explains variation in house prices.

**Higher R² is better.**

### RMSE

Root Mean Squared Error measures the average prediction error while giving larger errors more weight.

**Lower RMSE is better.**

### MAE

Mean Absolute Error represents the average absolute difference between actual and predicted prices.

**Lower MAE is better.**

## 📈 Visualizations

The project contains the following visualizations:

### 1. Average House Price by City

Shows how average property prices differ between cities.

### 2. House Price Distribution

Shows the distribution of property prices in the dataset.

### 3. House Price vs Area

Shows the relationship between property area and price.

### 4. Feature Importance

Shows the most important features used by the Random Forest model.

### 5. Actual vs Predicted Prices

Compares actual house prices with the prices predicted by the Random Forest model.

A prediction closer to the diagonal reference line indicates better model performance.

---

## 🔮 Example Prediction

The trained model is also used to predict the price of a new property.

Example property:

```text
Area      : 10 Marla
Bedrooms  : 4
Bathrooms : 4
City      : Islamabad
Type      : House
```

The model generates an estimated price in Pakistani Rupees (PKR).

> The predicted value is an estimate produced by the machine learning model and should not be treated as an exact market price.

---

## 📌 Key Findings

The project compares Linear Regression and Random Forest Regression based on their test performance.

The final model is selected automatically according to the higher **test R² score**.

Feature importance from Random Forest is also used to identify which property characteristics contribute most strongly to the model's predictions.

---

## 👩‍💻 Author

**Urooj Fatima**

Data Science Internship — Month 2 

**Project:** House Price Prediction

---

## 📜 Internship Requirement

This project was completed as part of the **CloudExify Data Science Month 2 — Project 3: House Price Prediction** internship task.

