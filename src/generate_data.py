"""
Generates a synthetic loan dataset with the SAME schema as the public
"Loan Prediction" dataset (Analytics Vidhya / Kaggle), so you can swap in the
real CSV later without changing any other code.

Usage:  python src/generate_data.py
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 3000

gender = rng.choice(["Male", "Female"], N, p=[0.81, 0.19])
married = rng.choice(["Yes", "No"], N, p=[0.65, 0.35])
dependents = rng.choice(["0", "1", "2", "3+"], N, p=[0.57, 0.17, 0.16, 0.10])
education = rng.choice(["Graduate", "Not Graduate"], N, p=[0.78, 0.22])
self_emp = rng.choice(["No", "Yes"], N, p=[0.86, 0.14])
prop_area = rng.choice(["Urban", "Semiurban", "Rural"], N, p=[0.33, 0.38, 0.29])

base = rng.lognormal(mean=8.35, sigma=0.55, size=N)
base *= np.where(education == "Graduate", 1.25, 0.9)
base *= np.where(self_emp == "Yes", 1.2, 1.0)
income = np.clip(base, 1500, 80000).round()

co_income = np.where(rng.random(N) < np.where(married == "Yes", 0.65, 0.2),
                     rng.lognormal(7.3, 0.6, N), 0).clip(0, 40000).round()

loan_amt = (income + 0.5 * co_income) * rng.uniform(0.018, 0.075, N) + rng.normal(0, 12, N)
loan_amt = np.clip(loan_amt, 9, 700).round()
term = rng.choice([120, 180, 240, 300, 360, 480], N, p=[0.02, 0.07, 0.02, 0.03, 0.82, 0.04])
credit = (rng.random(N) < 0.84).astype(float)

# Hidden approval rule (credit history dominates, as in the real data)
total_inc = income + co_income
emi_ratio = (loan_amt * 1000 / term) / total_inc
score = (
    3.2 * (credit - 0.5)
    - 3.0 * (emi_ratio - 0.28)
    + 0.35 * (education == "Graduate")
    + 0.30 * (prop_area == "Semiurban")
    + 0.20 * (married == "Yes")
    - 0.10 * (dependents == "3+")
    + 0.25 * np.log(total_inc / 5000)
    + rng.normal(0, 0.9, N)
)
status = np.where(score > 0.55, "Y", "N")

df = pd.DataFrame({
    "Loan_ID": [f"LP{100000 + i}" for i in range(N)],
    "Gender": gender, "Married": married, "Dependents": dependents,
    "Education": education, "Self_Employed": self_emp,
    "ApplicantIncome": income, "CoapplicantIncome": co_income,
    "LoanAmount": loan_amt, "Loan_Amount_Term": term.astype(float),
    "Credit_History": credit, "Property_Area": prop_area, "Loan_Status": status,
})

# Realistic missing values (the real dataset has these too)
for col, p in [("Gender", .02), ("Married", .005), ("Dependents", .025),
               ("Self_Employed", .05), ("LoanAmount", .035),
               ("Loan_Amount_Term", .02), ("Credit_History", .08)]:
    df.loc[rng.random(N) < p, col] = np.nan

df.to_csv("data/loan_data.csv", index=False)
print(df.shape, "| approval rate:", (df.Loan_Status == "Y").mean().round(3))
