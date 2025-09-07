# Rainfall Prediction using Machine Learning

A machine learning project that predicts rainfall based on meteorological data using Random Forest classification with hyperparameter tuning.

## 🌧️ Project Overview

This project implements a rainfall prediction system that analyzes various weather parameters to determine whether it will rain or not. The model uses Random Forest classification with GridSearchCV for hyperparameter optimization to achieve optimal performance.

## 📊 Dataset

The dataset contains meteorological features including:
- **Pressure**: Atmospheric pressure
- **Temperature**: Maximum, minimum, and current temperature
- **Dewpoint**: Dew point temperature
- **Humidity**: Relative humidity percentage
- **Cloud**: Cloud coverage
- **Sunshine**: Hours of sunshine
- **Wind**: Wind speed and direction
- **Rainfall**: Target variable (yes/no)

## 🔧 Technologies Used

- **Python 3.x**
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Machine learning algorithms and tools
- **Pickle**: Model serialization

## 📁 Project Structure

```
rainfall-prediction/
│
├── DataSets/
│   └── Rainfall.csv
├── rainfall_prediction_model.pkl
├── Rainfall_Prediction_Using_ML.py
└── README.md
```

## 🚀 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Rohan-GitHub-A/Rainfall_Prediction_Using_ML.git
   cd rainfall-prediction
   ```

2. **Install required packages**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

3. **Run the project**
   ```bash
   python Rainfall_Prediction_Using_ML.ipynb
   ```

## 📈 Model Development Process

### 1. Data Collection & Processing
- Loaded meteorological data from CSV file
- Handled missing values using mode (categorical) and median (numerical)
- Converted target variable from text ("yes"/"no") to binary (1/0)
- Removed unnecessary columns and whitespace

### 2. Exploratory Data Analysis (EDA)
- Generated distribution plots for all numerical features
- Created correlation heatmap to identify feature relationships
- Analyzed target variable distribution
- Used boxplots to detect outliers

### 3. Data Preprocessing
- Removed highly correlated features (maxtemp, temperature, mintemp)
- Handled class imbalance using downsampling technique
- Split data into training (80%) and testing (20%) sets

### 4. Model Training
- Implemented Random Forest Classifier
- Performed hyperparameter tuning using GridSearchCV
- Optimized parameters:
  - `n_estimators`: [50, 100, 200]
  - `max_features`: ["sqrt", "log2"]
  - `max_depth`: [None, 10, 20, 30]
  - `min_samples_split`: [2, 5, 10]
  - `min_samples_leaf`: [1, 2, 4]

### 5. Model Evaluation
- Used 5-fold cross-validation for robust performance assessment
- Evaluated model using accuracy, confusion matrix, and classification report
- Saved the best model using pickle for future predictions

## 🎯 Model Performance

The model achieved excellent performance with:
- **Cross-validation accuracy**: High consistency across folds
- **Test set performance**: Detailed metrics available in classification report
- **Balanced predictions**: Effective handling of class imbalance

## 💡 Usage Example

```python
import pickle
import pandas as pd

# Load the trained model
with open("rainfall_prediction_model.pkl", "rb") as file:
    model_data = pickle.load(file)

model = model_data["model"]
feature_names = model_data["feature_names"]

# Make prediction with new data
input_data = (1015.9, 19.9, 95, 81, 0.0, 40.0, 13.7)
input_df = pd.DataFrame([input_data], columns=feature_names)

prediction = model.predict(input_df)
result = "Rainfall" if prediction[0] == 1 else "No Rainfall"
print(f"Prediction result: {result}")
```

## 📊 Features Used for Prediction

The final model uses these meteorological features:
- Pressure
- Dewpoint
- Humidity
- Cloud coverage
- Sunshine hours
- Wind direction
- Wind speed

## 🔍 Key Insights

- Temperature-related features showed high correlation and were removed to prevent multicollinearity
- Class imbalance was successfully addressed through downsampling
- Random Forest proved effective for this meteorological prediction task
- Hyperparameter tuning significantly improved model performance

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- Scikit-learn documentation and community
- Weather prediction research papers and methodologies

## 📧 Contact

Your Name - [rohanku2111@gmail.com](rohanku2111@gmail.com)

Project Link: https://github.com/Rohan-GitHub-A/Rainfall_Prediction_Using_ML

---

⭐ If you found this project helpful, please give it a star!
