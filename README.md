# 📦 Demand Forecasting for Meal Delivery  

## 📝 Project Overview  
This project aims to forecast demand for a meal delivery company operating in multiple cities. The company has several fulfillment centers responsible for dispatching meal orders to customers. Accurate demand prediction is crucial for optimizing stock management, reducing waste, and improving efficiency.  

### 🎯 Objective  
Predict the number of orders (`num_orders`) for the next 10 weeks (Weeks: 136-145) for each **center-meal** combination in the test dataset.  

## 📊 Dataset Description  
The dataset consists of multiple files:  

- **train.xlsx**: Historical data of meal orders  
- **test.xlsx**: Data to be used for prediction  
- **meal_info.csv**: Information about meals (category, cuisine)  
- **fulfillment_center_info.csv**: Data on fulfillment centers (city, region, type, operation area)  

### 🔍 Key Features  
| Feature | Description |
|---------|------------|
| `id` | Unique identifier for each record |
| `week` | Week number |
| `center_id` | Fulfillment center ID |
| `meal_id` | Unique meal ID |
| `checkout_price` | Final meal price after discounts |
| `base_price` | Original meal price before discounts |
| `emailer_for_promotion` | Whether the meal was promoted via email (1 = Yes, 0 = No) |
| `homepage_featured` | Whether the meal was featured on the homepage (1 = Yes, 0 = No) |
| `num_orders` | Number of orders (target variable) |
| `category` | Meal category (e.g., Beverages, Snacks, etc.) |
| `cuisine` | Cuisine type (e.g., Indian, Thai, Continental) |
| `city_code` | City identifier |
| `region_code` | Region identifier |
| `center_type` | Type of fulfillment center |
| `op_area` | Operational area of the center |

## 🔍 Methodology  
### 1️⃣ Data Preprocessing  
- Handling missing values and duplicates  
- Merging datasets (`meal_info` and `fulfillment_center_info`)  
- One-hot encoding categorical variables  
- Splitting data into **training (80%)** and **testing (20%)**  

### 2️⃣ Exploratory Data Analysis (EDA)  
- Visualizing order trends over weeks  
- Checking feature distributions and correlations  
- Identifying outliers in meal pricing and demand  

### 3️⃣ Model Development  
The following models are used for forecasting:  
1. **LightGBM** 🚀 (Primary model due to efficiency and speed)  
2. **XGBoost** 🌲 (Gradient boosting for robust performance)  
3. **Random Forest** 🌳 (Baseline ensemble method)  
4. **Ensemble - Averaging** 🎛️ (Combining multiple models)  
5. **Stacking** 🔗 (Layering models for improved accuracy)  

### 4️⃣ Model Evaluation  
- Performance metrics: **RMSE, MAE, MAPE**  
- Feature importance analysis to determine key drivers of demand  

## 📌 Results & Insights  
- **Best Model**: Stacking with RMSE = **163.5543** and R² = **83.82%**  
- **Feature Importance**: `checkout_price`, `op_area`, and `homepage_featured` are key drivers of demand.  
- **Ensemble vs. Single Models**: Stacking improved accuracy by ~5% compared to individual models.
