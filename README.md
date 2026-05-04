# CHAID Model Pipeline

## Overview
This project contains a Jupyter Notebook demonstrating a Chi-square Automatic Interaction Detection (CHAID) Decision Tree model. The pipeline predicts a gym member's `Experience_Level` using the `chefboost` library. Unlike standard CART models, CHAID requires categorical features, so this pipeline emphasizes continuous-to-categorical data binning and type casting.

## Prerequisites
* **Python:** Version 3.8 or higher is recommended.
* **Graphviz (System Level):** The visualization cell relies on Graphviz. You must install the Graphviz core engine on your OS (e.g., `sudo apt install graphviz` on Ubuntu, `brew install graphviz` on macOS, or via the [official installer for Windows](https://graphviz.org/download/)) in addition to the Python package.

## Dependencies
The following Python libraries are required to execute this machine learning pipeline:
- `pandas`: For data loading, binning, and manipulation.
- `matplotlib`: For generating plots and visualizations.
- `scikit-learn`: For data splitting and model evaluation metrics (Confusion Matrix, Classification Report).
- `chefboost`: The core framework used to build and train the CHAID decision tree.
- `graphviz`: For rendering the structural flowchart of the decision tree.
- `jupyter` (or `notebook`): To open and execute the `.ipynb` file.

## Installation
You can install all the required Python dependencies by running the following command in your terminal or command prompt:
```bash
pip install pandas matplotlib scikit-learn chefboost graphviz jupyter
```

## Data Setup
Ensure your folder structure looks exactly like this before running the notebook. The dataset must be inside a Dataset folder relative to the notebook.
```bash
├── your_chaid_notebook.ipynb
└── Dataset/
    └── gym_members_exercise_tracking.csv
```
(Note: During execution, Chefboost will automatically create an outputs/rules/ directory to store the generated Python rules.py file containing the tree logic.)

## How to Run the Pipeline
1. Download the files to your local machine, keeping the directory structure intact.
2. Open a terminal/command prompt and navigate to the folder containing the notebook.
3. Launch Jupyter Notebook by executing: `jupyter notebook`
4. In the web interface, click on the notebook file to open it.
5. Run the notebook sequentially, cell by cell (using `Shift + Enter`), or select **Kernel > Restart & Run All** from the top menu.

## Pipeline Structure
The notebook guides you through the following phases:
1. **Import Libraries & Load Data**: Loads the CSV dataset and displays the initial records.
2. **Preprocessing & Splitting (The CHAID Fixes)**: 
   - **Binning:** Converts continuous variables (e.g., Age, BMI, Calories_Burned) into 3 evenly distributed categories (`Low`, `Medium`, `High`) using `pd.qcut`.
   - **String Formatting:** Ensures frequency is a distinct string and casts the target variable to a string.
   - Splits the data into **Training (60%)**, **Validation (20%)**, and **Testing (20%)** sets.
3. **Prepare Data for Chefboost**: Renames the target column strictly to `Decision` and forces the entire DataFrame to the `object` data type to prevent math/division errors inside Chefboost.
4. **Build CHAID Model**: Configures and fits the `Chefboost` model using the CHAID algorithm.
5. **Testing & Evaluation**: Iterates over the test set (converting instances to standard Python lists to avoid Pandas index errors), generates predictions, and prints the Accuracy and Classification Report.
6. **Visualizations**: Displays a Confusion Matrix and builds a custom top-to-bottom flowchart using `graphviz` to map the tree's logic visually.
7. **Final Audit**: Performs automated sanity checks validating that the `rules.py` file was created, predictions are contiguous (no `None` values), and accuracy boundaries are intact.
