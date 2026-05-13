# Gym Members Exercise Classification using Decision Trees

Machine Learning project focused on predicting the workout type of gym members using Decision Tree algorithms, hyperparameter optimization, and dataset analysis techniques.

Repository:
https://github.com/WonderLandOfAndrew/Decision-Trees-Project

---

## Overview

This project uses the Gym Members Exercise Dataset from Kaggle in order to predict the workout type of a sportsperson based on physiological, demographic, and workout-related attributes.

The project was developed as part of a Machine Learning assignment focused on Decision Trees and supervised classification algorithms. The implementation includes dataset analysis, preprocessing, hyperparameter tuning, visualization techniques, and model evaluation.

Dataset source:
https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset

---

## Objectives

The main objectives of the project are:

- Analyze the gym members dataset
- Predict workout types using Decision Tree algorithms
- Improve model accuracy through hyperparameter tuning
- Perform exploratory data analysis (EDA)
- Evaluate classification performance using multiple metrics
- Visualize dataset insights and model behavior

---

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Dataset Description

The dataset contains information about gym members and their exercise habits, including:

- Age
- Gender
- Weight
- Height
- BMI
- Calories Burned
- Heart Rate
- Session Duration
- Water Intake
- Workout Frequency
- Experience Level
- Fat Percentage
- Workout Type (target variable)

The target variable represents the type of training performed by the athlete.

---

## Machine Learning Workflow

The project follows a complete machine learning pipeline:

1. Data loading and inspection
2. Exploratory Data Analysis (EDA)
3. Data preprocessing
4. Encoding categorical variables
5. Train/test split
6. Decision Tree model training
7. Hyperparameter optimization
8. Model evaluation
9. Results visualization

---

## Decision Tree Hyperparameter Tuning

Different hyperparameters were tested in order to maximize model performance and reduce overfitting.

The tuned parameters include:

- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `criterion`
- `max_leaf_nodes`

Different combinations of these parameters were evaluated to achieve the highest possible accuracy.

---

## Model Evaluation

The model performance was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The evaluation process helped identify the strengths and limitations of the Decision Tree classifier.

---

## Data Analysis and Visualization

The project also includes dataset analysis and visualizations such as:

- Workout type distribution
- Correlation heatmaps
- Feature importance analysis
- Decision Tree visualization
- Statistical summaries
- Confusion matrices

These visualizations help better understand the dataset structure and the model behavior.

---

## Repository Structure

```text
Decision-Trees-Project/
│
├── dataset/
├── notebooks/
├── images/
├── models/
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/WonderLandOfAndrew/Decision-Trees-Project.git
```

Move into the project folder:

```bash
cd Decision-Trees-Project
```

Create a virtual environment:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the notebook and run the cells sequentially.

---

## Possible Improvements

Future improvements may include:

- Random Forest implementation
- Gradient Boosting algorithms
- Cross-validation techniques
- Feature selection methods
- Neural Network comparison
- Web application deployment

---

## Educational Purpose

This project was developed for educational and academic purposes in order to better understand:

- Decision Tree algorithms
- Classification techniques
- Hyperparameter optimization
- Exploratory Data Analysis
- Machine Learning workflows

---

## References

- https://scikit-learn.org/1.5/modules/tree.html
- https://scikit-learn.org/1.5/api/sklearn.tree.html
- https://www.datacamp.com/tutorial/decision-tree-classification-python
- https://www.geeksforgeeks.org/decision-tree/
- https://github.com/PPStef/invatare-automata-lab/tree/master/3.%20Arbori%20de%20decizie

---

## Author

Developed as part of a Machine Learning laboratory project focused on Decision Tree classification techniques and predictive analytics in sports and fitness datasets.
