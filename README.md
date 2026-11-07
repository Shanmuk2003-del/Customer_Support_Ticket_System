# Financial Customer Complaint Classification and Routing System

A beginner-friendly text-classification application for financial consumer complaints. It uses a saved machine-learning model to predict a CFPB product category, maps that category to a department, applies separate keyword-based priority rules, and provides a Streamlit interface for viewing and managing tickets.

> **Scope:** The model was trained on financial consumer complaints from the Consumer Financial Protection Bureau (CFPB). These are not general customer-support tickets, and the project should not be treated as a production-ready case-management system.

## Problem statement

Financial complaint text can be difficult to sort consistently. This project demonstrates how text classification can suggest a product category and how business rules can route and prioritize a complaint. It also provides a small local interface for saving tickets and reviewing basic trends.

## Key features

- Predicts one of eight CFPB product categories from complaint text.
- Shows the predicted category, a model confidence estimate, and scores for the available categories.
- Routes the predicted category to a configured department.
- Assigns High, Medium, or Low priority using keyword rules (not machine learning).
- Stores submitted tickets in a local SQLite database.
- Displays ticket history, supports searching and filtering, and allows status changes.
- Shows dashboard counts by status, category, department, and priority, plus a daily creation trend.

## Technology stack

- Python
- pandas and NumPy for dataset handling
- scikit-learn for TF-IDF features, classification pipelines, and evaluation
- joblib for loading and saving fitted pipelines
- Streamlit for the web interface
- SQLite for local ticket storage

Pinned runtime dependencies are listed in `requirements.txt`.

## Project architecture

```text
Complaint text
    |
    v
Streamlit interface (app.py)
    |
    v
Processing workflow (process_complaint.py)
    |                       |
    v                       v
Saved Logistic        Rule-based department
Regression model      mapping and priority rules
(predict_complaint.py)       (routing.py)
    |                       |
    +-----------+-----------+
                v
        Prediction result
          /           \
         v             v
SQLite ticket store   Dashboard and ticket views
(db.py)               (app.py)
```

The application loads a previously fitted Logistic Regression pipeline. Starting the app does not train or retrain a model.

## Dataset

The project uses the CFPB consumer complaints dataset. The repository's extracted dataset contains **78,313 records**. After text cleaning, category selection, and removal of records with conflicting labels, the processed modeling dataset contains **20,122 records across eight product categories**.

The eight retained categories are:

1. Bank account or service
2. Checking or savings account
3. Credit card
4. Credit card or prepaid card
5. Credit reporting, credit repair services, or other personal consumer reports
6. Debt collection
7. Money transfer, virtual currency, or money service
8. Mortgage

The data represents financial consumer complaints; it is not a representative sample of all customer-support requests.

## Machine-learning approach

Complaint text is transformed into TF-IDF features, then classified with two compared models:

- **Logistic Regression**
- **Linear Support Vector Machine (Linear SVM)**

The TF-IDF setup considers individual words and two-word phrases, removes common English stop words, and applies document-frequency limits. The vectorizer is fitted within each pipeline using training data only.

### Evaluation results

The saved models were evaluated on the held-out test split. Metrics below are percentages.

| Metric | Logistic Regression | Linear SVM |
|---|---:|---:|
| Accuracy | 72.56% | 73.68% |
| Macro Precision | 67.66% | 70.36% |
| Macro Recall | 69.97% | 66.68% |
| Macro F1 | **68.65%** | 68.01% |
| Weighted F1 | 72.71% | 72.85% |

Logistic Regression was selected for the application because it has the higher **Macro F1** score. It was **not** selected for higher overall accuracy; Linear SVM has higher accuracy and weighted F1 in this evaluation. Macro F1 gives each category equal weight when combining precision and recall, which is useful when comparing performance across multiple categories.

## Department routing

`routing.py` maps product categories to departments:

| Product category | Department |
|---|---|
| Credit card or prepaid card; Credit card | Cards and Payments |
| Checking or savings account; Bank account or service | Deposit Accounts |
| Mortgage | Mortgage Services |
| Credit reporting, credit repair services, or other personal consumer reports | Credit Reporting |
| Debt collection | Debt Collection |
| Money transfer, virtual currency, or money service | Money Transfers |

Unrecognized categories return an Unknown department result rather than an invented mapping.

## Rule-based priority assignment

Priority is assigned by case-insensitive keyword checks in `routing.py`; it is **not a trained machine-learning model**. High-priority terms are checked first, followed by Medium and then Low. If no configured term matches, or the input is empty or invalid, the default is Medium.

This is a simple heuristic: it does not understand context, intent, negation, or the full meaning of a complaint. A matching word alone does not establish the true urgency of a case.

## Streamlit interface and SQLite ticket management

Run the app locally to use its complaint form and dashboard. A submitted complaint is processed and can be saved as a ticket with an automatically generated ID, an ISO-formatted UTC creation time, and an initial status of **Open**.

The interface includes:

- Recent ticket history.
- Search by ticket ID or partial complaint description.
- Exact filters for category, department, priority, and status; filters can be combined.
- Status management for **Open**, **In Progress**, and **Resolved**.
- Dashboard totals and grouped counts, plus a daily ticket creation trend.

The SQLite database is `data/complaints.db`. It is local storage, not a hosted or multi-user database. The daily trend uses SQLite's UTC date handling and an inclusive cutoff date, so it is a calendar-date filter rather than an exact rolling 30-by-24-hour window.

## Installation and startup

Run these commands from the project directory in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The startup command uses the existing `app.py` Streamlit application. The app expects the project files, including the saved model under `models/`, to be available relative to the project directory.

## Project directory structure

```text
Customer_Support_Ticket_System/
├── app.py
├── db.py
├── predict_complaint.py
├── process_complaint.py
├── routing.py
├── train_models.py
├── split_dataset.py
├── prepare_cfpb_dataset.py
├── extract_json_data.py
├── analyze_cfpb_categories.py
├── analyze_text_quality.py
├── check_cfpb_quality.py
├── check_data_quality.py
├── inspect_categories.py
├── inspect_dataset.py
├── inspect_new_dataset.py
├── investigate_duplicates.py
├── verify_processed_dataset.py
├── requirements.txt
├── data/
│   ├── complaints.db
│   ├── raw/
│   │   ├── customer_support_tickets.csv
│   │   └── cfpb/
│   │       ├── complaints-2021-05-14_08_16_.json
│   │       └── complaints_extracted.csv
│   └── processed/
│       ├── cfpb_model_ready.csv
│       ├── train.csv
│       └── test.csv
├── models/
│   ├── logistic_regression_pipeline.joblib
│   └── linear_svm_pipeline.joblib
├── docs/
│   └── PROJECT_DOCUMENTATION.md
├── notebooks/
├── reports/
├── screenshots/
└── src/
```

The CFPB JSON file is normalized into an extracted CSV; the processed dataset and train/test splits are separate files. The standalone `customer_support_tickets.csv` is present in the repository, but the documented CFPB model workflow uses the CFPB files under `data/raw/cfpb/` and `data/processed/`.

## Testing summary

The recorded project testing summary is **47 passed, 0 failed, 0 not tested**.

## Known limitations

- The model is specific to the selected CFPB financial complaint categories and may perform poorly on general support requests or unfamiliar wording.
- Priority is a keyword heuristic and can miss context or assign an unsuitable priority.
- Confidence estimates are model scores, not calibrated probabilities or guarantees of correctness.
- The dashboard's “Last 30 Days” view uses an inclusive UTC date cutoff rather than a rolling time window; dates near midnight may differ from a user's local date.
- SQLite is local to the project and has no built-in authentication, role management, or multi-user deployment support in this application.
- The interface displays complaint text stored in the database. Do not use real or sensitive personal information in an unprotected demonstration environment.

## Future improvements

- Evaluate on newer CFPB data and on carefully designed out-of-domain examples.
- Calibrate confidence estimates and expose uncertainty more clearly.
- Improve priority rules or evaluate a separate, validated prioritization method.
- Add suitable authentication, authorization, privacy controls, and retention policies before any broader deployment.
- Consider a managed database and operational monitoring if the project is extended beyond a local demonstration.

## Responsible use and privacy

Use synthetic complaint descriptions for demonstrations and testing. Real financial complaints may contain personal or sensitive information; do not enter, publish, or share such data without appropriate authorization and safeguards. Treat predictions and priorities as decision-support suggestions that require human review. This educational project is not production-ready and does not provide authenticated access, automatic email notifications, or live external integrations.

For a fuller technical explanation, see [docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md).
