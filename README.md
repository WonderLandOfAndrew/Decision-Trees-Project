# Data Setup

The notebook expects the dataset to be placed in a specific directory relative to the notebook file.

Ensure your folder structure looks exactly like this before running the notebook:

```
├── CART_Model_Pipeline.ipynb
└── Dataset/
    └── gym_members_exercise_tracking.csv
```

(Note: If the dataset is missing or named differently, the notebook will throw a FileNotFoundError at the data loading step.)

# How to Run the Pipeline
1. Download the files to your local machine, keeping the directory structure intact.
2. Open a terminal/command prompt and navigate to the folder containing CART_Model_Pipeline.ipynb.
3. Launch Jupyter Notebook by executing: `jupyter notebook`
4. In the web interface that opens, click on CART_Model_Pipeline.ipynb.
5. Run the notebook sequentially, cell by cell (using Shift + Enter), or select Kernel > Restart & Run All from the top menu.

# Pipeline Structure
The notebook guides you through the following Machine Learning steps:
1. Import Libraries & Load Data: Loads the CSV dataset and previews the records.
2. Preprocessing & Splitting:
    - Encodes categorical string variables (like Gender).
    - Splits the data into Training (60%), Validation (20%), and Testing (20%) sets.
3. Build & Validate CART Model: Initializes a DecisionTreeClassifier (using Gini impurity and max_depth=6) and evaluates its initial accuracy on the Validation set.
4. Testing & Evaluation Pipeline: Tests the finalized model on the unseen Testing set, printing out the overall Test Accuracy and a detailed Classification Report (Precision, Recall, F1-Score).
5. Final Audit & Visualizations: Performs an automated test pipeline validation, plots the logic of the decision tree, and outputs a Confusion Matrix display.
