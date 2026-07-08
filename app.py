import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Loan Risk Assessment",
    page_icon="💰",
    layout="centered"
)

# =========================================================
# LOAD THE TRAINED MODEL
# =========================================================
MODEL_PATH = "naive_bayes_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)
    except Exception:
        model = None

# The exact column order the model was trained on.
# (Falls back to this fixed list if the model object doesn't expose it.)
FEATURE_ORDER = [
    "person_age", "person_income", "person_emp_exp", "loan_amnt",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
    "credit_score", "previous_loan_defaults_on_file", "person_gender_male",
    "person_education_Bachelor", "person_education_Doctorate",
    "person_education_High School", "person_education_Master",
    "person_home_ownership_OTHER", "person_home_ownership_OWN",
    "person_home_ownership_RENT", "loan_intent_EDUCATION",
    "loan_intent_HOMEIMPROVEMENT", "loan_intent_MEDICAL",
    "loan_intent_PERSONAL", "loan_intent_VENTURE",
]
if model is not None and hasattr(model, "feature_names_in_"):
    FEATURE_ORDER = list(model.feature_names_in_)

# =========================================================
# QUESTION DEFINITIONS
# Plain-language help text for every field, since applicants may not know
# banking/finance terminology.
# =========================================================
QUESTIONS = [
    {
        "key": "person_age",
        "question": "🎂 What is your age?",
        "help": "Enter your current age in completed years.",
        "type": "number", "min": 18, "max": 100, "default": 30, "step": 1,
    },
    {
        "key": "person_gender",
        "question": "⚧ What is your gender?",
        "help": "Used only as one of the statistical factors in the risk model.",
        "type": "radio", "options": ["Female", "Male"],
    },
    {
        "key": "person_education",
        "question": "🎓 What is your highest completed education level?",
        "help": (
            "**High School** = secondary school. **Associate** = a 2-year college diploma. "
            "**Bachelor** = a standard 3–4 year undergraduate degree. **Master** = a postgraduate "
            "degree after Bachelor's. **Doctorate** = the highest research degree (PhD)."
        ),
        "type": "select",
        "options": ["High School", "Associate", "Bachelor", "Master", "Doctorate"],
    },
    {
        "key": "person_income",
        "question": "💵 What is your annual income (before tax)?",
        "help": "Your total yearly income from all sources, before any taxes are deducted, "
                "in your local currency.",
        "type": "number", "min": 1000, "max": 10_000_000, "default": 50000, "step": 1000,
    },
    {
        "key": "person_emp_exp",
        "question": "💼 How many years of work experience do you have?",
        "help": "Total number of years you have been employed in your working life so far.",
        "type": "number", "min": 0, "max": 60, "default": 3, "step": 1,
    },
    {
        "key": "person_home_ownership",
        "question": "🏠 What is your current home ownership status?",
        "help": (
            "**RENT** = you pay rent to a landlord. **OWN** = you own your home outright. "
            "**MORTGAGE** = you are still paying off a home loan on your property. "
            "**OTHER** = any other living arrangement (e.g. living with family, free of charge)."
        ),
        "type": "select", "options": ["RENT", "MORTGAGE", "OWN", "OTHER"],
    },
    {
        "key": "loan_amnt",
        "question": "🏦 How much money do you want to borrow (loan amount)?",
        "help": "The total amount you are requesting to borrow, in your local currency.",
        "type": "number", "min": 500, "max": 100000, "default": 10000, "step": 500,
    },
    {
        "key": "loan_intent",
        "question": "🎯 What is the purpose of this loan?",
        "help": (
            "**PERSONAL** = general personal expenses. **EDUCATION** = tuition/study costs. "
            "**MEDICAL** = medical/health expenses. **VENTURE** = starting or growing a business. "
            "**HOMEIMPROVEMENT** = renovating/repairing your home. "
            "**DEBTCONSOLIDATION** = combining multiple existing debts into a single new loan."
        ),
        "type": "select",
        "options": ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE",
                    "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
    },
    {
        "key": "loan_int_rate",
        "question": "📈 What interest rate has been offered/quoted for this loan (%)?",
        "help": "The annual interest rate for this loan, shown as a percentage "
                "(for example, enter 12.5 for 12.5%). Check your loan offer document if unsure.",
        "type": "number", "min": 1.0, "max": 40.0, "default": 11.0, "step": 0.1,
    },
    {
        "key": "cb_person_cred_hist_length",
        "question": "📜 How many years is your credit history?",
        "help": "'cb' stands for Credit Bureau — the agency that tracks your borrowing history. "
                "This is the number of years since you opened your very first credit account "
                "(loan, credit card, etc.).",
        "type": "number", "min": 0, "max": 40, "default": 4, "step": 1,
    },
    {
        "key": "credit_score",
        "question": "💳 What is your credit score?",
        "help": "A number (commonly between 300–850, like a FICO score) that lenders use to judge "
                "how risky it is to lend you money. Higher = considered safer. "
                "You can usually find this on a credit report from a credit bureau.",
        "type": "number", "min": 300, "max": 850, "default": 650, "step": 1,
    },
    {
        "key": "previous_loan_defaults_on_file",
        "question": "⚠️ Do you have any previous loan defaults on record?",
        "help": "A 'default' means a past loan or credit payment that was not repaid as agreed. "
                "Answer 'Yes' if your credit report/credit bureau file shows any such record.",
        "type": "radio", "options": ["No", "Yes"],
    },
]

TOTAL_STEPS = len(QUESTIONS)

# =========================================================
# SESSION STATE
# =========================================================
if "step" not in st.session_state:
    st.session_state.step = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "finished" not in st.session_state:
    st.session_state.finished = False


def go_back():
    st.session_state.step -= 1


def restart():
    st.session_state.step = 0
    st.session_state.answers = {}
    st.session_state.finished = False


# =========================================================
# HEADER
# =========================================================
st.title("💰 Loan Risk Assessment")
st.markdown(
    "Answer a few simple questions about yourself and the loan you're applying for. "
    "If a term is unfamiliar, read the explanation shown below each question."
)
st.divider()

# =========================================================
# CONVERSATIONAL Q&A FLOW
# =========================================================
if not st.session_state.finished:

    step = st.session_state.step
    st.progress(step / TOTAL_STEPS, text=f"Question {min(step + 1, TOTAL_STEPS)} of {TOTAL_STEPS}")

    q = QUESTIONS[step]
    st.subheader(q["question"])
    input_key = f"input_{q['key']}"

    if q["type"] == "number":
        is_float = isinstance(q["step"], float)
        answer = st.number_input(
            "Your answer",
            min_value=q["min"], max_value=q["max"],
            value=st.session_state.answers.get(q["key"], q["default"]),
            step=q["step"],
            label_visibility="collapsed",
            key=input_key,
        )
    elif q["type"] == "radio":
        prev = st.session_state.answers.get(q["key"])
        default_index = q["options"].index(prev) if prev in q["options"] else 0
        answer = st.radio(
            "Your answer", q["options"], index=default_index,
            label_visibility="collapsed", key=input_key,
        )
    elif q["type"] == "select":
        prev = st.session_state.answers.get(q["key"])
        default_index = q["options"].index(prev) if prev in q["options"] else 0
        answer = st.selectbox(
            "Your answer", q["options"], index=default_index,
            label_visibility="collapsed", key=input_key,
        )

    st.caption(f"ℹ️ {q['help']}")

    col1, col2 = st.columns([1, 1])
    with col1:
        if step > 0:
            st.button("⬅️ Back", on_click=go_back, use_container_width=True)
    with col2:
        def save_and_next():
            st.session_state.answers[q["key"]] = st.session_state[input_key]
            if step + 1 == TOTAL_STEPS:
                st.session_state.finished = True
            else:
                st.session_state.step += 1

        button_label = "Check My Risk ✅" if step + 1 == TOTAL_STEPS else "Next ➡️"
        st.button(button_label, on_click=save_and_next, use_container_width=True, type="primary")

# =========================================================
# RESULTS
# =========================================================
else:
    a = st.session_state.answers

    income = max(a.get("person_income", 1), 1)  # avoid divide-by-zero
    loan_amnt = a.get("loan_amnt", 0)
    loan_percent_income = round(loan_amnt / income, 4)

    # Build the one-hot / label-encoded feature row exactly as the model expects
    row = {col: 0 for col in FEATURE_ORDER}
    row["person_age"] = a.get("person_age", 30)
    row["person_income"] = income
    row["person_emp_exp"] = a.get("person_emp_exp", 0)
    row["loan_amnt"] = loan_amnt
    row["loan_int_rate"] = a.get("loan_int_rate", 10.0)
    row["loan_percent_income"] = loan_percent_income
    row["cb_person_cred_hist_length"] = a.get("cb_person_cred_hist_length", 0)
    row["credit_score"] = a.get("credit_score", 600)
    row["previous_loan_defaults_on_file"] = 1 if a.get("previous_loan_defaults_on_file") == "Yes" else 0

    if a.get("person_gender") == "Male" and "person_gender_male" in row:
        row["person_gender_male"] = 1

    edu = a.get("person_education")
    edu_col = f"person_education_{edu}"
    if edu_col in row:
        row[edu_col] = 1

    home = a.get("person_home_ownership")
    home_col = f"person_home_ownership_{home}"
    if home_col in row:
        row[home_col] = 1

    intent = a.get("loan_intent")
    intent_col = f"loan_intent_{intent}"
    if intent_col in row:
        row[intent_col] = 1

    st.subheader("📊 Your Result")

    with st.expander("📋 See a summary of your answers"):
        st.write(f"- **Age:** {a.get('person_age')}")
        st.write(f"- **Gender:** {a.get('person_gender')}")
        st.write(f"- **Education:** {a.get('person_education')}")
        st.write(f"- **Annual income:** {income:,.0f}")
        st.write(f"- **Work experience:** {a.get('person_emp_exp')} years")
        st.write(f"- **Home ownership:** {a.get('person_home_ownership')}")
        st.write(f"- **Loan amount requested:** {loan_amnt:,.0f}")
        st.write(f"- **Loan purpose:** {a.get('loan_intent')}")
        st.write(f"- **Interest rate:** {a.get('loan_int_rate')}%")
        st.write(f"- **Loan as % of income:** {loan_percent_income * 100:.1f}%")
        st.write(f"- **Credit history length:** {a.get('cb_person_cred_hist_length')} years")
        st.write(f"- **Credit score:** {a.get('credit_score')}")
        st.write(f"- **Previous defaults on file:** {a.get('previous_loan_defaults_on_file')}")

    if model is None:
        st.error(
            "⚠️ The prediction model could not be loaded. "
            "Please make sure 'naive_bayes_model.pkl' is placed in the app folder."
        )
    else:
        try:
            X = pd.DataFrame([[row[col] for col in FEATURE_ORDER]], columns=FEATURE_ORDER)
            prediction = model.predict(X)

            st.divider()
            if prediction[0] == 1:
                st.error("⚠️ Based on the information provided, this loan shows a **higher risk of default**.")
                st.info("This is only a general indicator. A lender would review your full application "
                        "and documents before making any final decision.")
            else:
                st.success("✅ Based on the information provided, this loan shows a **low risk of default**.")
                st.info("This is only a general indicator, not a loan approval. "
                        "Actual approval depends on the lender's full review process.")

            st.caption(
                "🔒 Disclaimer: This is an automated, general-purpose estimate for informational "
                "purposes only. It is not a loan approval or financial advice — please consult "
                "your bank or a financial advisor for an official decision."
            )
        except Exception:
            st.error(
                "Something went wrong while processing your answers. "
                "Please check that all fields were filled in correctly and try again."
            )

    st.divider()
    st.button("🔄 Start Over", on_click=restart)
