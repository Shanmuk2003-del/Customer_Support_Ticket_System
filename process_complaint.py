"""
process_complaint.py

END-TO-END COMPLAINT PROCESSING PIPELINE

This script combines three steps into one workflow:
  1. Classify the complaint -> product category (ML model)
  2. Route to a department         (business rule)
  3. Assign a priority            (business rule)

It REUSES the existing functions from:
  - predict_complaint.py  -> predict_complaint()
  - routing.py            -> recommend_department() and assign_priority()

No model is retrained here. The saved Logistic Regression pipeline is loaded
once and reused for every prediction.
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
# Reuse the existing functions instead of duplicating their logic.
# ---------------------------------------------------------------------------
from predict_complaint import predict_complaint
from routing import recommend_department, assign_priority


# ---------------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------------
def process_complaint(complaint_text):
    """
    Run the full complaint processing pipeline.

    Parameters
    ----------
    complaint_text : str
        The raw complaint description.

    Returns
    -------
    result : dict
        A structured dictionary containing:
          - complaint_text
          - predicted_category
          - confidence_estimate
          - top_categories
          - department
          - department_reason
          - priority
          - priority_reason
    """
    # Step 1: Predict the product category (ML model)
    predicted_category, confidence, all_scores = predict_complaint(complaint_text)

    # Step 2: Route to a department (business rule)
    department, department_reason = recommend_department(predicted_category)

    # Step 3: Assign priority (business rule)
    priority, priority_reason = assign_priority(complaint_text)

    return {
        "complaint_text": complaint_text,
        "predicted_category": predicted_category,
        "confidence_estimate": confidence,
        "top_categories": all_scores,
        "department": department,
        "department_reason": department_reason,
        "priority": priority,
        "priority_reason": priority_reason,
    }


# ---------------------------------------------------------------------------
# Helper to print a nicely formatted result
# ---------------------------------------------------------------------------
def print_result(result, index):
    print("\n" + "=" * 70)
    print(f"COMPLAINT #{index}")
    print("=" * 70)
    print(f"Complaint      : \"{result['complaint_text']}\"")
    print(f"Category       : {result['predicted_category']}")
    print(f"Confidence     : {result['confidence_estimate']:.4f} "
          f"({result['confidence_estimate'] * 100:.1f}%)")
    print(f"Department     : {result['department']}")
    print(f"Priority       : {result['priority']}")
    print(f"Priority reason: {result['priority_reason']}")
    print("\nTop category scores (model confidence estimates, not guaranteed probabilities):")
    for category, score in result["top_categories"][:4]:
        bar = "#" * int(score * 40)
        print(f"    {category:<55} {score:.4f}  {bar}")


# ---------------------------------------------------------------------------
# Test with five synthetic complaint examples
#    (NOT real personal narratives - these are generic examples we wrote)
# ---------------------------------------------------------------------------
TEST_EXAMPLES = [
    # High priority: fraud-related
    "I noticed an unauthorized transaction on my credit card and I think my card was stolen.",
    # Medium priority: fee/billing dispute
    "I was charged an unexpected fee on my checking account and I want to dispute it.",
    # Low priority: general information request
    "I have a general question about how my savings account works.",
    # Default case: no keyword matches
    "I am writing about my mortgage and need some help with the paperwork.",
    # High priority: identity theft
    "I believe I am a victim of identity theft and someone opened an account in my name.",
]

print("\n" + "=" * 70)
print("END-TO-END COMPLAINT PROCESSING")
print("=" * 70)

for i, complaint in enumerate(TEST_EXAMPLES, start=1):
    try:
        result = process_complaint(complaint)
        print_result(result, i)
    except ValueError as e:
        print(f"\nComplaint #{i} - INPUT REJECTED: {e}")
    except Exception as e:
        print(f"\nComplaint #{i} - UNEXPECTED ERROR: {e}")

print("\n" + "=" * 70)
print("IMPORTANT NOTES")
print("=" * 70)
print("- Product category: predicted by the Logistic Regression ML model.")
print("- Confidence estimates: model scores, NOT guaranteed probabilities.")
print("- Department: assigned by a business rule (routing.py).")
print("- Priority: assigned by keyword rules (routing.py), NOT a trained model.")
print("  Priority is a heuristic based on keywords like 'fraud', 'unauthorized',")
print("  'fee', 'dispute', etc. It is NOT a machine-learned prediction.")
print("=" * 70)