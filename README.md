# 💰 Loan Risk Assessment App

A Streamlit app that asks a loan applicant simple, plain-language questions
one at a time, then uses your trained Gaussian Naive Bayes model
(`naive_bayes_model.pkl`) to estimate the risk that the loan would default.

## How the questions map to your model

Your model was trained on the `loan_data.csv` dataset with these exact
22 features (confirmed directly from `model.feature_names_in_`):

| Model feature | Asked as |
|---|---|
| `person_age` | Age |
| `person_income` | Annual income |
| `person_emp_exp` | Years of work experience |
| `loan_amnt` | Loan amount requested |
| `loan_int_rate` | Interest rate quoted (%) |
| `loan_percent_income` | **Auto-calculated** = loan amount ÷ income |
| `cb_person_cred_hist_length` | Years of credit history |
| `credit_score` | Credit score (300–850) |
| `previous_loan_defaults_on_file` | Previous defaults on record? (Yes/No) |
| `person_gender_male` | Gender (Male/Female) |
| `person_education_*` | Education level (High School / Associate / Bachelor / Master / Doctorate) |
| `person_home_ownership_*` | Home ownership (RENT / MORTGAGE / OWN / OTHER) |
| `loan_intent_*` | Purpose of loan (PERSONAL / EDUCATION / MEDICAL / VENTURE / HOMEIMPROVEMENT / DEBTCONSOLIDATION) |

The one-hot columns (education, home ownership, loan intent) and their
"dropped" baseline category (Associate / MORTGAGE / DEBTCONSOLIDATION) exactly
match how the notebook (`naivebayesproject.ipynb`) encoded the training data
with `pd.get_dummies(..., drop_first=True)`.

**Note on the target label:** In the notebook, `loan_status = 1` means the
loan **defaulted**, and `0` means it did **not** default — the app's
"higher risk" / "low risk" messages follow this same convention.

## Files needed in the same folder

- `app.py` — the Streamlit app
- `naive_bayes_model.pkl` — your trained model file
- `requirements.txt` — Python dependencies

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the link shown in the terminal (usually `http://localhost:8501`).

## Deploy for free on Streamlit Community Cloud

1. Create a GitHub repository and upload these 3 files:
   - `app.py`
   - `requirements.txt`
   - `naive_bayes_model.pkl`
2. Go to https://share.streamlit.io and sign in with your GitHub account.
3. Click **"New app"**, select your repository/branch, and set the main file
   path to `app.py`.
4. Click **Deploy** — you'll get a public URL to share.

Any time you push new changes to GitHub, the deployed app updates automatically.

## Notes on privacy & content

- The app never shows model accuracy, training metrics, or any internal/
  technical details — only a simple low-risk / higher-risk message with a
  short disclaimer.
- No answers are stored or sent anywhere; everything happens only in the
  current browser session.
- Every question includes a plain-language explanation so applicants
  unfamiliar with financial terms (credit score, credit history, debt
  consolidation, mortgage, etc.) can still answer confidently.
- Clearly labeled as a general **risk indicator**, not an official loan
  approval or financial advice.

## Customizing

- To adjust question wording, ranges, or explanations, edit the `QUESTIONS`
  list near the top of `app.py`.
- The app automatically reads `model.feature_names_in_` from your pickle
  file to determine the exact column order — if you retrain the model with
  different features, the app will adapt automatically as long as the
  question keys still line up with the new feature names.
