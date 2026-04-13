## Project Overview
This project is a Jupyter Notebook (`loan.ipynb`) that solves a classic binary classification problem — predicting loan approval status (`Loan_Status`) based on historical bank customer data. 

The model analyzes various borrower parameters such as income, credit history, marital status, and other characteristics to determine whether a loan should be granted.

## Tech Stack
* Python (Core programming language)
* Pandas / NumPy (Data processing, cleaning, and analysis)
* Seaborn / Matplotlib (Data visualization)
* Scikit-Learn (Data preprocessing, building and evaluating machine learning models)

## Dataset
The project uses a dataset containing the following key customer features:
* Loan_ID — Unique application identifier
* Gender, Married, Dependents — Gender, marital status, and number of dependents
* Education, Self_Employed — Education level and employment status
* ApplicantIncome, CoapplicantIncome — Income of the applicant and co-applicant
* LoanAmount, Loan_Amount_Term — Requested loan amount and loan term
* Credit_History — Credit history record (1.0 — good, 0.0 — poor)
* Property_Area — Type of property area (Urban, Rural, Semiurban)
* `Loan_Status` — Loan approval status (Y — Approved, N — Rejected) — Target Variable

## Main Project Steps
1. Exploratory Data Analysis (EDA):
   - Analyzing the shape and structure of the dataset.
   - Identifying missing values.
   - Building visualizations (using `seaborn`) to show the distribution of borrowers by gender, dependents, employment type, etc.

2. Data Preprocessing:
   - Handling Missing Values: Categorical features are filled with the most frequent value (mode), while numerical features (e.g., `LoanAmount`) are filled with the mean.
   - Feature Engineering: Created a new feature for total family income (`FamIncome` = ApplicantIncome + `CoapplicantIncome`).
   - Normalization: Applied logarithmic transformation (`np.log`) to incomes and loan amounts to reduce the impact of outliers and normalize the distribution.

3. Preparation for Machine Learning:
   - Transforming categorical values into numerical ones using LabelEncoder.
   - Splitting the data into a feature matrix (`X`) and the target variable (`y`).
   - Standardizing features using StandardScaler.

4. Model Training and Evaluation:
   The project tests and compares several classic machine learning algorithms:
   - K-Nearest Neighbors (KNN) — Achieved the highest accuracy on the test set (~75%)
   - Random Forest Classifier — Accuracy ~70%
   - Decision Tree Classifier — Accuracy ~66.6%
   - Gaussian Naive Bayes (GaussianNB)

## How to Run the Project
1. Clone the repository or download the loan.ipynb file to your local machine.
2. Install the required libraries (if not already installed): 
   ```bash
   pip install pandas numpy seaborn scikit-learn jupyter
