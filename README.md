# Financial Customer Complaint Classification and Routing System

A beginner-friendly text-classification application for financial consumer complaints. It uses a saved machine-learning model to predict a CFPB product category, maps that category to a department, applies separate keyword-based priority rules, and provides a Streamlit interface for viewing and managing tickets.

**Core components:**
- **TF-IDF** text features with **Logistic Regression** and **Linear SVM** comparison.
- **Rule-based priority** assignment (High / Medium / Low) using keyword checks.
- **Department routing** that maps predicted categories to business departments.
- **Streamlit** web interface for prediction, ticket management, search, filtering, and dashboards.
- **SQLite** local ticket storage.

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

## Dataset acquisition

The modeling dataset originates from the CFPB consumer complaints dataset available on Kaggle:

https://www.kaggle.com/datasets/venkatasubramanian/automatic-ticket-classification

**The raw JSON file is not included in this GitHub repository.** To reproduce the full pipeline, download the dataset and place the JSON file at the following path before running the preparation scripts:

```
data/raw/cfpb/complaints-2021-05-14_08_16_.json
```

All data files, trained models, and the SQLite database are excluded from version control by `.gitignore`. See the [Important artifact warning](#important-artifact-warning) section below.

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

### Prerequisites

- **Python 3.14** (verified in the project's tested environment; `python --version` should report `Python 3.14.x`).
- **Git** (already installed if you are cloning the repository).
- **Windows PowerShell** (all commands below are written for PowerShell).

The pinned runtime dependencies are declared in `requirements.txt`:

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | 1.67.0 | Web interface (`app.py`) |
| `pandas` | 3.1.1 | Dataset handling |
| `scikit-learn` | 1.9.1 | TF-IDF, classification, evaluation |
| `joblib` | 1.6.0 | Model serialization |
| `numpy` | 2.5.4 | Numerical arrays (`train_models.py`) |

### Setup steps

Run these commands from the project directory in Windows PowerShell:

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install pinned dependencies
python -m pip install -r requirements.txt
```

**Two different workflows are described below.** Do not assume that a fresh GitHub clone can run both.

#### Workflow A: Run the existing application

After completing steps 1 and 2, start the application:

```powershell
streamlit run app.py
```

This workflow requires the trained model at `models/logistic_regression_pipeline.joblib`. The application loads the model at startup and does **not** retrain anything. If the model file is missing, submitting a complaint will fail.

#### Workflow B: Reproduce the dataset preparation and training pipeline

This workflow also requires:

- The two required preparation scripts listed in the [Project directory structure](#project-directory-structure) section: `extract_json_data.py` and `prepare_cfpb_dataset.py`.
- The raw CFPB JSON dataset, downloaded from Kaggle and placed at `data/raw/cfpb/complaints-2021-05-14_08_16_.json`.

After obtaining these, run the scripts in the following order. Each stage reads the output of the previous stage.

```powershell
# 1. Extract nested CFPB complaint data from the raw JSON
python extract_json_data.py
# Output: data/raw/cfpb/complaints_extracted.csv

# 2. Clean text, filter categories, remove conflicting labels
python prepare_cfpb_dataset.py
# Output: data/processed/cfpb_model_ready.csv

# 3. Create a leakage-safe 80/20 train/test split
python split_dataset.py
# Output: data/processed/train.csv
#         data/processed/test.csv

# 4. Train and compare models, then save fitted pipelines
python train_models.py
# Output: models/logistic_regression_pipeline.joblib
#         models/linear_svm_pipeline.joblib
```

**Stage 1** normalizes the nested `_source` field from the raw JSON into a flat CSV.

**Stage 2** cleans complaint text, keeps only categories with at least 500 records (eight categories remain), removes records whose text is associated with conflicting labels, and keeps only `complaint_id`, `complaint_what_happened`, and `product`.

**Stage 3** uses `StratifiedGroupKFold` with 5 folds so identical complaint descriptions are never split across training and testing. It selects the fold closest to an 80/20 split and verifies that zero descriptions are shared between the two files.

**Stage 4** fits a `TfidfVectorizer` plus classifier inside each pipeline, evaluates on the held-out test split, and saves both fitted pipelines with `joblib`.

## Important artifact warning

The following items are **excluded from this GitHub repository** by `.gitignore`:

- Raw CFPB JSON and extracted CSV (`data/raw/`)
- Processed dataset and train/test splits (`data/processed/`)
- Trained models (`models/*.joblib`)
- SQLite database (`data/complaints.db`, matched by `*.db`)
- Python virtual environment (`.venv/`)
- Python cache directories (`__pycache__/`, `*.pyc`)

**To run the application**, you need the trained model at `models/logistic_regression_pipeline.joblib`. Without it, submitting a complaint will fail. If you only want to run the existing application, obtaining the fitted model file is sufficient. **To reproduce the full dataset preparation and training pipeline**, you must also obtain the raw CFPB JSON dataset and the two required preparation scripts (`extract_json_data.py` and `prepare_cfpb_dataset.py`) listed in the [Project directory structure](#project-directory-structure) section.

## Project directory structure

The tree below distinguishes files **tracked in GitHub** from files **generated locally** during dataset preparation and model training.

### Tracked in GitHub (committed source and documentation)

The following 12 paths are the exact files returned by `git ls-files`:

```text
Customer_Support_Ticket_System/
├── .gitignore
├── README.md
├── requirements.txt
├── app.py
├── db.py
├── predict_complaint.py
├── process_complaint.py
├── routing.py
├── split_dataset.py
├── train_models.py
├── verify_processed_dataset.py
└── docs/
    └── PROJECT_DOCUMENTATION.md
```

### Untracked scripts (present locally, not published)

The following 10 scripts exist in the current working copy but are **not** included in the published GitHub repository. They fall into two groups:

**Required for dataset preparation and training** (2 scripts):

- `extract_json_data.py`
- `prepare_cfpb_dataset.py`

**Optional analysis and inspection utilities** (8 scripts, not needed to execute the training pipeline):

- `analyze_cfpb_categories.py`
- `analyze_text_quality.py`
- `check_cfpb_quality.py`
- `check_data_quality.py`
- `inspect_categories.py`
- `inspect_dataset.py`
- `inspect_new_dataset.py`
- `investigate_duplicates.py`

A fresh clone of the repository contains only the 12 tracked files listed above. To run the full preparation and training pipeline, you must obtain the two required scripts (`extract_json_data.py` and `prepare_cfpb_dataset.py`) from the local working copy, or add them to version control separately. The eight optional scripts are not needed to train models or run the application.

### Generated locally (excluded from GitHub by `.gitignore`)

```text
Customer_Support_Ticket_System/
├── .venv/                    # Virtual environment
├── __pycache__/              # Python bytecode cache
├── data/
│   ├── complaints.db         # SQLite ticket store
│   ├── raw/
│   │   ├── customer_support_tickets.csv   # Legacy local CSV, excluded from GitHub
│   │   └── cfpb/
│   │       ├── complaints-2021-05-14_08_16_.json   # Must be downloaded from Kaggle
│   │       └── complaints_extracted.csv             # Generated by extract_json_data.py
│   └── processed/
│       ├── cfpb_model_ready.csv                  # Generated by prepare_cfpb_dataset.py
│       ├── train.csv                             # Generated by split_dataset.py
│       └── test.csv                              # Generated by split_dataset.py
├── models/
│   ├── logistic_regression_pipeline.joblib        # Generated by train_models.py
│   └── linear_svm_pipeline.joblib                 # Generated by train_models.py
├── notebooks/                # Empty
├── reports/                  # Empty
├── screenshots/              # Empty
└── src/                      # Empty
```

The CFPB JSON file is normalized into an extracted CSV; the processed dataset and train/test splits are separate files. The standalone `customer_support_tickets.csv` under `data/raw/` is a legacy local file that is excluded from GitHub by `.gitignore`. The documented CFPB model workflow uses the CFPB files under `data/raw/cfpb/` and `data/processed/`.

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
