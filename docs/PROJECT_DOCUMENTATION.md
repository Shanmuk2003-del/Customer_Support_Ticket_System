# Project Documentation

## 1. Introduction and objectives

The Financial Customer Complaint Classification and Routing System is a local demonstration application for organizing financial consumer complaint text. Its main objectives are to:

- Predict a CFPB product category from a complaint description.
- Route the predicted category to a configured department.
- Apply a separate rule-based priority label.
- Save and review tickets through a Streamlit interface backed by SQLite.

The training data comes from the Consumer Financial Protection Bureau (CFPB) consumer complaints dataset. It represents financial complaints, not general customer-support tickets. The project is intended for learning and demonstration, not production use.

## 2. End-to-end workflow

1. A user enters a complaint description in the Streamlit app (`app.py`).
2. On submission, `process_complaint.py` calls `predict_complaint.py` to classify the text using the saved Logistic Regression pipeline.
3. The predicted product category is passed to the department mapping in `routing.py`.
4. The complaint text is separately checked against priority keywords in `routing.py`.
5. The app displays the category, confidence estimate, department, priority, and priority reason.
6. The app attempts to save the processed result in `data/complaints.db` through `db.py`.
7. The history, filtering, status-management, and dashboard sections read from the same SQLite database.

The saved model is loaded from disk. Running the app does not run the training script or retrain either model.

## 3. Dataset extraction and preprocessing

The CFPB source file is a JSON dataset at `data/raw/cfpb/complaints-2021-05-14_08_16_.json`. `extract_json_data.py` reads the JSON records, normalizes their nested `_source` fields into tabular columns, and writes `data/raw/cfpb/complaints_extracted.csv`.

The extracted dataset contains **78,313 records**. The project then creates `data/processed/cfpb_model_ready.csv`, which contains **20,122 records**. That processed modeling dataset has three columns:

- `complaint_id`
- `complaint_what_happened`
- `product`

## 4. Data cleaning and category selection

`prepare_cfpb_dataset.py` performs the documented preparation steps:

1. Fill missing complaint descriptions with empty strings, convert descriptions to strings, and trim surrounding whitespace.
2. Remove descriptions that are empty after trimming.
3. Retain product categories with at least 500 records in the cleaned data.
4. Find complaint descriptions associated with more than one product label and remove records with those conflicting descriptions.
5. Keep the complaint ID, complaint text, and product label for the processed modeling dataset.

The resulting data has eight product labels:

- Bank account or service
- Checking or savings account
- Credit card
- Credit card or prepaid card
- Credit reporting, credit repair services, or other personal consumer reports
- Debt collection
- Money transfer, virtual currency, or money service
- Mortgage

This selection reduces the original CFPB product-label set to categories with enough examples for this project. It also means the model cannot predict categories outside the retained set.

## 5. Duplicate complaint handling

Repeated text can lead to overly optimistic evaluation if an identical description is present in both training and test data. The preparation step removes descriptions that have conflicting product labels, but it does not generally remove all repeated same-label descriptions.

`split_dataset.py` uses the full complaint description as the group key. As a result, identical descriptions are assigned to a single side of the split rather than being divided between training and testing. This is the duplicate-handling step that prevents text overlap across the evaluation split.

## 6. Group-aware train/test split and leakage prevention

`split_dataset.py` uses scikit-learn's `StratifiedGroupKFold` with five folds, shuffling, and a fixed random seed of 42. It selects the fold whose test-set size is closest to 20% of the processed data, giving an approximately 80/20 train/test split.

- **Stratified** aims to keep product-category proportions similar across the split.
- **Group-aware** keeps identical complaint descriptions together.

Without grouping, a repeated description could appear in both training and testing. A model could then benefit from having effectively seen the test example during training, making its evaluation less representative of unseen text.

The current split contains 16,098 training records and 4,024 test records. `train.csv` and `test.csv` are stored under `data/processed/`.

## 7. TF-IDF in simple terms

TF-IDF converts text into numbers that a classifier can use. It gives more weight to words or phrases that are useful for distinguishing documents and less weight to terms that appear broadly across many documents.

For example, the synthetic phrase **“late payment fee”** can contribute features for the individual words and for two-word combinations such as “payment fee.” The vectorizer represents each complaint as a numerical feature vector rather than interpreting it as a person would.

The project configuration:

- Removes common English stop words.
- Uses unigrams and bigrams (`ngram_range=(1, 2)`).
- Ignores terms appearing in fewer than two documents (`min_df=2`).
- Ignores terms appearing in more than 95% of documents (`max_df=0.95`).
- Uses sublinear term-frequency scaling.

The vectorizer is part of each scikit-learn pipeline, so it is fitted on training text and then applied to held-out test text. This prevents vocabulary fitting from using test-set text.

## 8. Logistic Regression

Logistic Regression is a supervised classification algorithm. During training, it learns how the TF-IDF features relate to each of the product labels. For a new description, it scores the available classes and predicts one category.

In this project, the Logistic Regression classifier uses `class_weight="balanced"` and `max_iter=1000`. Its fitted TF-IDF-plus-classifier pipeline is saved as `models/logistic_regression_pipeline.joblib`. This is the model loaded for application predictions.

## 9. Linear SVM

A Linear Support Vector Machine (SVM) is another supervised text-classification method. It learns linear decision boundaries over the TF-IDF features to distinguish product categories.

The project trains a `LinearSVC` classifier with `class_weight="balanced"` and saves its pipeline as `models/linear_svm_pipeline.joblib`. It is used for model comparison; the application prediction module loads the Logistic Regression pipeline instead.

## 10. Model comparison and evaluation metrics

`train_models.py` builds and evaluates both models on the held-out test set. The currently saved pipelines were also evaluated against the repository's existing test split; the results match the table below. Metrics are shown as percentages.

| Metric | Logistic Regression | Linear SVM |
|---|---:|---:|
| Accuracy | 72.56% | 73.68% |
| Macro Precision | 67.66% | 70.36% |
| Macro Recall | 69.97% | 66.68% |
| Macro F1 | 68.65% | 68.01% |
| Weighted F1 | 72.71% | 72.85% |

Metric meanings:

- **Accuracy:** the share of all test examples assigned the correct category.
- **Precision:** among examples assigned to a category, how many truly belong to it.
- **Recall:** among examples in a category, how many the model finds.
- **F1:** a combined measure of precision and recall.
- **Macro average:** calculate a metric for each category and average the category scores equally.
- **Weighted average:** average category scores while weighting each category by its number of test examples.

## 11. Why Logistic Regression was selected

Linear SVM had higher overall accuracy (73.68% compared with 72.56%) and slightly higher weighted F1 (72.85% compared with 72.71%). Logistic Regression had the higher Macro F1 (68.65% compared with 68.01%) and was selected on that basis.

Macro F1 gives each category equal weight. Selecting by this metric prioritizes the average balance of precision and recall across categories, rather than choosing solely by overall accuracy. The selection is not a claim that Logistic Regression wins every metric or every future dataset.

## 12. Prediction pipeline

`predict_complaint.py` loads the fitted Logistic Regression pipeline from `models/logistic_regression_pipeline.joblib`. For a valid non-empty string, it:

1. Calls the pipeline to predict one product category.
2. Calls `predict_proba` to obtain the model's class scores.
3. Returns the highest score as the confidence estimate and a sorted list of all category scores.

`process_complaint.py` combines that prediction with department mapping and priority assignment, then returns a result dictionary used by the app. The prediction function rejects `None`, non-string values, empty strings, and whitespace-only strings with `ValueError`.

## 13. Confidence estimates and their limitations

The application displays the maximum score returned by the Logistic Regression pipeline as a confidence estimate. These scores are **not calibrated probabilities**. For example, a score of 0.8 does not establish that the prediction is correct 80% of the time.

Scores indicate how the fitted model distributes its output across its learned classes. They do not measure whether the complaint is actually in scope, whether a human reviewer would agree, or whether the model is reliable on a different population.

## 14. Department mapping logic

`recommend_department()` performs an exact lookup of the predicted product category in a fixed Python dictionary:

| Product category | Department |
|---|---|
| Credit card or prepaid card | Cards and Payments |
| Credit card | Cards and Payments |
| Checking or savings account | Deposit Accounts |
| Bank account or service | Deposit Accounts |
| Mortgage | Mortgage Services |
| Credit reporting, credit repair services, or other personal consumer reports | Credit Reporting |
| Debt collection | Debt Collection |
| Money transfer, virtual currency, or money service | Money Transfers |

An invalid or unmapped category returns an Unknown result with an explanation. This is deterministic rule-based routing, not another trained model.

## 15. Priority assignment keyword rules

Priority is assigned by `assign_priority()` in `routing.py`. It lowercases the description and searches for configured substrings, checking the High group first, then Medium, then Low:

- **High:** `fraud`, `unauthorized transaction`, `identity theft`, `account takeover`, `stolen money`, `stole money`, `stolen card`, `lost card`, `unauthorized access`
- **Medium:** `incorrect charge`, `billing dispute`, `payment problem`, `overdraft fee`, `interest dispute`, `delayed refund`, `unexpected fee`, `incorrect balance`
- **Low:** `general question`, `information request`, `account information`, `how do i`, `routine service`, `need information`, `account inquiry`

If none of the terms match, the function returns Medium as the default. Empty or invalid text also returns Medium with a reason indicating that the default was used.

Priority is assigned using keyword-based business rules, **not a trained ML model**. Substring matching is not a full language-understanding system: context, negation, irony, urgency, or new wording can produce an inappropriate result.

## 16. SQLite database schema and operations

`db.py` stores tickets in `data/complaints.db` in a `complaints` table with these columns:

| Column | Purpose |
|---|---|
| `id` | Auto-incrementing ticket ID and primary key |
| `complaint_text` | Submitted description |
| `predicted_category` | Model-predicted CFPB product category |
| `confidence_estimate` | Highest model score; not a calibrated probability |
| `department` | Mapped department |
| `priority` | Rule-based priority |
| `priority_reason` | Explanation of the matched keyword or default |
| `created_at` | UTC ISO-8601 creation timestamp |
| `status` | Ticket status, initially `Open` |

The module initializes the table if necessary, inserts a processed complaint, retrieves recent complaints, applies optional filters, updates a ticket status, and calculates dashboard aggregates. Allowed statuses are **Open**, **In Progress**, and **Resolved**. An update to a nonexistent positive ticket ID returns `False`; invalid status or non-positive ticket ID is rejected.

## 17. Streamlit application workflow

`app.py` is the application's entry point. It initializes the database, displays dashboard analytics, accepts a complaint description, and offers ticket-history and ticket-management sections. On a user-initiated prediction, it calls `process_complaint()`, displays the result, and attempts to save it to SQLite. Merely starting the app does not submit a complaint.

The app's layout is a local web interface served by Streamlit. The project does not implement user authentication, external API endpoints, automatic email notifications, or live external integrations.

## 18. Search, filtering, status management, and dashboard

The ticket interface supports:

- Recent history sorted by descending ticket ID.
- Partial complaint-description matching and exact matching for ticket ID.
- Exact filters for predicted category, department, priority, and status.
- Combining multiple filters; selecting “All” omits that filter.
- Updating a ticket to Open, In Progress, or Resolved.
- Clearing the current search filters.

The dashboard computes totals for all stored tickets and groups counts by status, category, department, and priority. It also groups creation counts by day for rows whose `date(created_at)` is greater than or equal to SQLite's `date('now', '-30 days')`.

Timestamps are stored in UTC. The dashboard comparison uses UTC dates and an inclusive calendar-date cutoff, not an exact rolling period of 30 days. The displayed set can therefore cover the cutoff date plus today, and the UTC day can differ from a user's local day.

## 19. Project testing and validation

The recorded project testing summary is **47 passed, 0 failed, 0 not tested**.

In addition, read-only evaluation of the existing saved pipelines against `data/processed/test.csv` reproduced the model metrics in Section 10. Database verification and synthetic input checks are separate validation activities; they do not train models or require inserting demo tickets into the existing database.

## 20. Current limitations and future improvements

### Current limitations

- The training data is CFPB financial complaint text; it does not represent every kind of support request or every current financial product.
- Only eight product categories are modeled.
- Text preprocessing and linear classifiers may not capture context, nuanced phrasing, or changes in how people describe issues.
- Priority is a keyword heuristic and may assign an unsuitable label.
- Confidence estimates are not calibrated probabilities.
- The local SQLite database and interface have no application authentication or role-based access controls.
- The daily dashboard trend is based on an inclusive UTC calendar-date cutoff, not a rolling 30-day duration.
- Complaint descriptions may contain sensitive information; the application does not provide a complete privacy or retention-management system.

### Possible future improvements

- Evaluate with newer CFPB data and clearly defined out-of-domain tests.
- Examine per-category errors and improve the handling of underrepresented labels.
- Add confidence calibration and a review path for uncertain predictions.
- Replace or augment priority keyword rules only after defining and evaluating suitable business criteria.
- Add appropriate authentication, authorization, audit, retention, and privacy controls before broader use.
- Consider managed storage, backups, and monitoring if extending beyond a local demonstration.

These are possible future directions; they are not currently implemented features.
