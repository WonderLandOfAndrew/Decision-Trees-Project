# Gym Members Exercise Dataset

## Overview
This repository uses the **Gym Members Exercise Dataset** available on Kaggle, which contains information about gym members, their physical characteristics, workout habits, and fitness-related metrics. The dataset is designed for machine learning and data analysis tasks focused on predicting workout types and understanding exercise behavior patterns.

Dataset source:  
https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset

---

# Dataset Description

The **Gym Members Exercise Dataset** is a structured fitness-related dataset designed to simulate real-world gym member information and workout monitoring data. It combines demographic information, biometric measurements, cardiovascular indicators, exercise habits, and physical performance metrics in order to provide a comprehensive view of an individual's training profile.

This dataset is highly suitable for:
- Supervised machine learning tasks
- Classification problems
- Predictive analytics
- Health and fitness analysis
- Exploratory data analysis
- Decision tree experimentation
- Feature importance evaluation

The data reflects common characteristics tracked in modern fitness environments such as gyms, smart fitness applications, wearable devices, and personal health monitoring systems.

---

# Dataset Characteristics

| Property | Description |
|---|---|
| Dataset Type | Tabular dataset |
| Domain | Fitness / Health / Machine Learning |
| Learning Task | Classification |
| Target Variable | `Workout_Type` |
| File Format | CSV |
| Data Type | Mixed numerical and categorical data |

The dataset includes both:
- Numerical features such as BMI, calories burned, heart rate, session duration, and water intake.
- Categorical features such as gender, workout type, and experience level.

This combination makes the dataset particularly useful for testing preprocessing pipelines involving:
- Encoding categorical variables
- Normalization
- Feature scaling
- Feature selection
- Train/validation/test splitting

---

# Understanding the Dataset Features

## Demographic Features

These attributes describe the personal profile of gym members.

| Feature | Explanation |
|---|---|
| Age | Represents the age of the participant. Age can influence exercise intensity, metabolism, and cardiovascular performance. |
| Gender | Indicates the participant's gender. This may impact physical performance metrics and workout preferences. |

---

## Physical and Body Measurements

These variables describe the body composition and physical characteristics of the gym members.

| Feature | Explanation |
|---|---|
| Weight (kg) | Body weight measured in kilograms. |
| Height (m) | Height measured in meters. |
| BMI | Body Mass Index calculated using weight and height. |
| Fat_Percentage | Estimated percentage of body fat. |

These features are important because they help determine:
- Physical condition
- Body composition
- Exercise efficiency
- Fitness level

---

## Cardiovascular Features

The dataset contains several heart rate measurements that provide insight into the participant’s physical exertion and cardiovascular condition.

| Feature | Explanation |
|---|---|
| Max_BPM | Maximum heart rate reached during exercise. |
| Avg_BPM | Average heart rate during the workout session. |
| Resting_BPM | Resting heart rate before physical activity. |

These features are highly relevant in exercise science because they can indicate:
- Training intensity
- Cardiovascular endurance
- Recovery capacity
- Exercise difficulty

Heart rate metrics are often strong predictors for workout classification models.

---

## Workout and Activity Features

These variables describe the participant’s training habits and exercise behavior.

| Feature | Explanation |
|---|---|
| Session_Duration (hours) | Total duration of the workout session. |
| Workout_Frequency (days/week) | Number of workout sessions performed weekly. |
| Experience_Level | Indicates how experienced the gym member is with training. |
| Workout_Type | The category/type of workout performed. This is the target variable used for prediction. |

The `Workout_Type` column represents the classification target for machine learning models. The goal of the project is to predict this value using all other features.

Possible workout categories may include:
- Cardio
- Strength
- HIIT
- Yoga
- Functional Training
- Mixed Workouts

---

## Lifestyle and Performance Features

These variables provide additional information regarding exercise efficiency and hydration.

| Feature | Explanation |
|---|---|
| Calories_Burned | Estimated amount of calories burned during the workout session. |
| Water_Intake (liters) | Daily water consumption measured in liters. |

These metrics help evaluate:
- Workout effectiveness
- Energy expenditure
- Hydration habits
- Physical performance

---

# Project Objective

The main goal of this project is to develop a machine learning model capable of predicting the type of workout performed by a gym member based on physiological and behavioral characteristics.

The project also includes:
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Feature analysis
- Model training and evaluation
- Hyperparameter tuning
- Performance comparison

---

# Machine Learning Approach

The project focuses on using **Decision Tree algorithms** and related ensemble methods.

Algorithms explored may include:
- ID3 (Iterative Dichotomiser 3)
- C4.5 Decision Tree
- CART (Classification and Regression Trees)
- CHAID

Different hyperparameters are tested to improve:
- Accuracy
- Precision
- Recall
- F1-Score

Examples of tuned hyperparameters:
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_leaf_nodes`
- `criterion`

---

# Exploratory Data Analysis

The dataset analysis may include:
- Distribution of workout types
- Correlation analysis
- BMI distribution
- Calories burned analysis
- Heart rate comparisons
- Feature importance analysis
- Missing value detection
- Class balancing analysis

Visualization examples:
- Histograms
- Boxplots
- Heatmaps
- Pie charts
- Confusion matrices

---

# Why This Dataset is Valuable

This dataset is valuable because it combines multiple dimensions of fitness analysis into a single structured dataset. Unlike simple datasets containing only a few biometric values, this dataset integrates:
- Health indicators
- Behavioral patterns
- Exercise metrics
- Performance measurements

This enables the development of more advanced machine learning models capable of identifying hidden relationships between:
- Body composition
- Cardiovascular activity
- Exercise intensity
- Workout preferences

The dataset is also suitable for educational purposes because it allows experimentation with:
- Decision Trees
- Random Forests
- Feature importance analysis
- Data preprocessing techniques
- Hyperparameter optimization
- Model evaluation metrics

---

# Potential Machine Learning Challenges

Several challenges may appear during analysis and model training.

## Feature Correlation
Some variables may be highly correlated, such as:
- Calories burned and session duration
- Weight and BMI
- Heart rate metrics

This may influence model behavior and feature importance.

---

## Class Imbalance
Certain workout types may appear more frequently than others, which can lead to:
- Biased predictions
- Reduced recall for minority classes

Techniques such as:
- Oversampling
- Undersampling
- Stratified splitting

may improve performance.

---

## Mixed Data Types
The dataset contains both:
- Numerical variables
- Categorical variables

Therefore preprocessing steps such as encoding are required before training machine learning models.

---

# Repository Structure

```bash
├── data/
│   └── gym_members_exercise_tracking.csv
│
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   ├── decision_tree_model.ipynb
│   └── hyperparameter_tuning.ipynb
│
├── results/
│   ├── charts/
│   ├── confusion_matrices/
│   └── trained_models/
│
├── README.md
└── requirements.txt
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/WonderLandOfAndrew/Decision-Trees-Project.git
```

Move into the project folder:

```bash
cd Decision-Trees-Project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

# Expected Outcomes

By completing this project, the following objectives are achieved:
- Better understanding of decision tree algorithms
- Experience with dataset preprocessing
- Hyperparameter optimization knowledge
- Model evaluation and interpretation
- Practical machine learning workflow implementation

---

# Educational Importance

This dataset represents an excellent real-world inspired example for studying:
- Classification algorithms
- Decision tree construction
- Data preprocessing
- Fitness analytics
- Predictive modeling
- Model interpretability

It is especially useful for understanding how machine learning can be applied in:
- Sports science
- Healthcare systems
- Smart fitness applications
- Wearable technology analytics
- Personalized training systems

---

# References

- Scikit-learn Decision Trees Documentation  
  https://scikit-learn.org/stable/modules/tree.html

- Scikit-learn API Reference  
  https://scikit-learn.org/stable/api/sklearn.tree.html

- Kaggle Dataset Source  
  https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset
