# 🏋️ Sportsman Training Time Prediction

A machine learning project that classifies gym workout session duration into categories (**Short**, **Medium**, **Long**) using a **C4.5-style Decision Tree** built with scikit-learn. The model is trained on real gym member exercise tracking data and walks through the full ML pipeline: exploratory analysis, feature engineering, hyperparameter tuning, and prediction.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Notebook Walkthrough](#-notebook-walkthrough)
- [Model Details](#-model-details)
- [Example Prediction](#-example-prediction)
- [Results](#-results)
- [Troubleshooting](#-troubleshooting)

---

## 🔍 Project Overview

The goal of this project is to predict how long a gym member's workout session will be, given features like age, heart rate, calories burned, workout type, and BMI. The session duration is bucketed into three classes:

| Class    | Duration              |
|----------|-----------------------|
| `Short`  | Less than 1.0 hour    |
| `Medium` | Between 1.0 – 1.5 hours |
| `Long`   | More than 1.5 hours   |

The classifier uses the **entropy criterion** (information gain), which is the core splitting strategy of the C4.5 / ID3 family of decision trees.

---

## 📦 Dataset

**File:** `gym_members_exercise_tracking.csv`

The dataset contains **15 columns** describing gym members and their exercise habits:

| Column | Description |
|--------|-------------|
| `Age` | Member's age |
| `Gender` | Male / Female |
| `Weight (kg)` | Body weight |
| `Height (m)` | Body height |
| `Max_BPM` | Maximum heart rate during workout |
| `Avg_BPM` | Average heart rate during workout |
| `Resting_BPM` | Resting heart rate |
| `Session_Duration (hours)` | ⭐ **Target variable** |
| `Calories_Burned` | Calories burned per session |
| `Workout_Type` | Type of workout (Cardio, Strength, etc.) |
| `Fat_Percentage` | Body fat percentage |
| `Water_Intake (liters)` | Daily water intake |
| `Workout_Frequency (days/week)` | How often per week |
| `Experience_Level` | Fitness experience (1 = beginner, 3 = advanced) |
| `BMI` | Body Mass Index |

> **Note:** Place the CSV file in the **same directory** as the notebook before running.

---

## ✅ Prerequisites

Make sure you have the following installed on your system:

- **Python 3.8 or higher** — [Download Python](https://www.python.org/downloads/)
- **pip** (comes bundled with Python)
- **Jupyter Notebook** or **JupyterLab** — to open and run the `.ipynb` file

To verify your Python version:
```bash
python --version
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/sportsman-training-time-prediction.git
cd sportsman-training-time-prediction
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv venv
```

Activate it:

- **Windows:**
  ```bash
  venv\Scripts\activate
  ```
- **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If a `requirements.txt` is not present, install the packages manually:

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

### 4. Launch Jupyter

```bash
jupyter notebook
```

Then open `sportsman_training_time_prediction.ipynb` from the Jupyter file browser.

---

## 📁 Project Structure

```
sportsman-training-time-prediction/
│
├── sportsman_training_time_prediction.ipynb   # Main notebook
├── gym_members_exercise_tracking.csv          # Dataset (place here)
├── requirements.txt                           # Python dependencies
└── README.md                                  # This file
```

---

## ▶️ How to Run

1. Make sure the CSV dataset file is in the **same folder** as the notebook.
2. Open the notebook in Jupyter.
3. Run all cells from top to bottom: **Kernel → Restart & Run All**

That's it — each section builds on the previous one, so running in order is important.

---

## 📓 Notebook Walkthrough

The notebook is organized into the following sections:

### 1. Imports & Version Check
Verifies that all required libraries are installed and prints their versions.

### 2. Load & Explore Dataset
Loads the CSV, prints shape, column names, missing values, and basic statistics.

### 3. Target Engineering
Creates the `Duration_Class` column by binning `Session_Duration (hours)` into `Short`, `Medium`, and `Long`. Visualizes the class distribution with a bar chart.

### 4. Feature Selection & Binning
Selects 8 meaningful features and discretizes continuous numeric columns (`Age`, `Avg_BPM`, `Calories_Burned`, `BMI`, `Workout_Frequency`) into categorical bins using `pd.cut`. This is required for the C4.5 / ID3 splitting strategy.

### 5. Label Encoding
Encodes all categorical columns (including the target) into integers using `LabelEncoder`, which scikit-learn requires.

### 6. Train / Test Split
Splits the data 80/20 with `stratify=y` to preserve class proportions.

### 7. Hyperparameter Exploration
Manually sweeps across:
- `max_depth` values: 2, 3, 4, 5, 6, 8, 10, None
- `min_samples_leaf` values: 1, 2, 4, 6, 8
- `min_samples_split` values: 2, 5, 10, 15, 20

Results are shown in tables to help you choose the best configuration.

### 8. Final Model Training
Trains the final `DecisionTreeClassifier` with `criterion="entropy"` and `max_depth=2`.

### 9. Evaluation
Prints accuracy, full classification report (precision, recall, F1 per class), and the confusion matrix.

### 10. Tree Visualization
Exports the decision rules as text and renders the tree graphically using `plot_tree`.

### 11. Prediction on New Data
Shows how to pass a new gym member's data through the pipeline to get a predicted duration class.

---

## 🌳 Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | `DecisionTreeClassifier` (scikit-learn) |
| Criterion | `entropy` (Information Gain — C4.5/ID3 style) |
| Splitter | `best` |
| Max Depth | `2` |
| Random State | `42` |
| Test Size | `20%` |

**Selected features used for training:**

- Age, Gender, Avg_BPM, Calories_Burned, Workout_Type, Workout_Frequency (days/week), Experience_Level, BMI

---

## 🔮 Example Prediction

To predict the session duration class for a new gym member, run the last cell of the notebook with your own values:

```python
sample = pd.DataFrame([{
    "Age": "Adult",
    "Gender": "Male",
    "Avg_BPM": "High_BPM",
    "Calories_Burned": "High_Cal",
    "Workout_Type": "Cardio",
    "Workout_Frequency (days/week)": "High_Freq",
    "Experience_Level": "2",
    "BMI": "Mid_BMI"
}])
```

The output will be one of: `Short`, `Medium`, or `Long`.

> **Tip:** Use the binning labels defined in the notebook (e.g., `"Young"/"Adult"/"Older"` for Age, `"Low_BPM"/"Mid_BPM"/"High_BPM"` for Avg_BPM, etc.)

---

## 📊 Results

The model is evaluated using:

- **Accuracy** — overall percentage of correct predictions
- **Classification Report** — per-class precision, recall, and F1-score
- **Confusion Matrix** — breakdown of predictions vs. actual classes

Refer to the notebook output cells for the exact numbers on your run.

---

## 🛠️ Troubleshooting

**`FileNotFoundError: gym_members_exercise_tracking.csv`**
→ Make sure the CSV file is in the **same directory** as the notebook.

**`ModuleNotFoundError: No module named 'sklearn'`**
→ Run `pip install scikit-learn` and restart the Jupyter kernel.

**Jupyter notebook won't open**
→ Make sure Jupyter is installed (`pip install jupyter`) and that you're running `jupyter notebook` from inside the project folder with the virtual environment activated.

**Plots not showing**
→ Add `%matplotlib inline` to the top of the notebook imports cell and re-run.

---

## 📄 License

This project is for educational purposes. Feel free to use and adapt it.
