"""
predict_complaint.py

Loads the already-trained Logistic Regression pipeline and uses it to
predict the product category for a NEW complaint description.

Key points:
  - The model is loaded from disk; NO retraining happens here.
  - Confidence estimates come from predict_proba, which are model
    confidence scores, NOT guaranteed probabilities.
  - Input validation rejects empty descriptions.
"""

import sys

# ---------------------------------------------------------------------------
# Dependency check: fail with a clear message instead of silently installing.
# ---------------------------------------------------------------------------
try:
    import joblib
except ImportError as exc:
    print("ERROR: The 'joblib' package is required but is not installed.")
    print("    pip install joblib")
    print(f"Original error: {exc}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 1. Load the pre-trained pipeline from disk.
#    We do NOT retrain here - we reuse the model saved by train_models.py.
# ---------------------------------------------------------------------------
PIPELINE_PATH = "models/logistic_regression_pipeline.joblib"
pipeline = joblib.load(PIPELINE_PATH)

# The class names are stored on the classifier inside the pipeline.
CLASS_NAMES = pipeline.named_steps["clf"].classes_.tolist()

print(f"Loaded pipeline from: {PIPELINE_PATH}")
print(f"Number of categories   : {len(CLASS_NAMES)}")
print(f"Categories             : {CLASS_NAMES}")


# ---------------------------------------------------------------------------
# 2. Prediction function with input validation
# ---------------------------------------------------------------------------
def predict_complaint(complaint_text):
    """
    Predict the product category for a complaint description.

    Parameters
    ----------
    complaint_text : str
        The raw complaint description.

    Returns
    -------
    predicted_category : str
        The predicted product category.
    confidence : float
        The model's confidence in the predicted category (0-1).
    all_scores : list of (category, score) tuples
        Confidence scores for every category, sorted descending.
    """
    # --- Input validation ---
    if complaint_text is None:
        raise ValueError("Input cannot be None.")
    if not isinstance(complaint_text, str):
        raise ValueError(f"Input must be a string, got {type(complaint_text).__name__}.")
    if complaint_text.strip() == "":
        raise ValueError("Complaint description cannot be empty.")

    # --- Predict category ---
    predicted_category = pipeline.predict([complaint_text])[0]

    # --- Confidence scores ---
    # These are model confidence estimates, NOT guaranteed probabilities.
    # They reflect how the model distributes its "belief" across categories.
    confidence_scores = pipeline.predict_proba([complaint_text])[0]
    confidence = float(confidence_scores.max())

    # Pair each category with its score and sort by score (descending)
    all_scores = sorted(
        zip(CLASS_NAMES, confidence_scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return predicted_category, confidence, all_scores


# ---------------------------------------------------------------------------
# 3. Test with three synthetic example complaints
#    (NOT real personal narratives - these are generic examples we wrote)
# ---------------------------------------------------------------------------
TEST_EXAMPLES = [
    "I was charged an overdraft fee on my checking account even though I had enough money in the account.",
    "My credit card was frozen without any warning and I cannot make payments on my credit card balance.",
    "I keep receiving calls from a debt collection agency about a debt I do not recognize and I want them to stop contacting me.",
]

print("\n" + "=" * 70)
print("PREDICTION EXAMPLES")
print("=" * 70)

for i, complaint in enumerate(TEST_EXAMPLES, start=1):
    try:
        predicted_category, confidence, all_scores = predict_complaint(complaint)
    except ValueError as e:
        print(f"\nExample {i} - INPUT REJECTED: {e}")
        continue

    print(f"\nExample {i}:")
    print(f"  Complaint : \"{complaint}\"")
    print(f"  Predicted category: {predicted_category}")
    print(f"  Confidence estimate: {confidence:.4f} ({confidence * 100:.1f}%)")
    print("  Confidence estimates for all categories (model scores, not guaranteed probabilities):")
    for category, score in all_scores:
        bar = "#" * int(score * 50)  # simple visual bar
        print(f"    {category:<55} {score:.4f}  {bar}")

print("\n" + "=" * 70)
print("Note: Confidence estimates are model scores that reflect how the model")
print("distributes its belief across categories. They are NOT guaranteed probabilities.")
print("=" * 70)