
# ==========================================
# FINANCIAL COMPLAINT ROUTING SYSTEM
# ==========================================


# PART 1: DEPARTMENT MAPPING

department_mapping = {
    "Credit card or prepaid card": "Cards and Payments",

    "Credit card": "Cards and Payments",

    "Checking or savings account": "Deposit Accounts",

    "Bank account or service": "Deposit Accounts",

    "Mortgage": "Mortgage Services",

    "Credit reporting, credit repair services, or other personal consumer reports":
        "Credit Reporting",

    "Debt collection": "Debt Collection",

    "Money transfer, virtual currency, or money service":
        "Money Transfers"
}


def recommend_department(product):

    if not isinstance(product, str) or not product.strip():
        return "Unknown", "Invalid product category"

    department = department_mapping.get(product)

    if department is None:
        return "Unknown", "No department mapping found"

    return department, "Mapped using product category"


# PART 2: PRIORITY ASSIGNMENT

high_priority_keywords = [
    "fraud",
    "unauthorized transaction",
    "identity theft",
    "account takeover",
    "stolen money",
    "stole money",
    "stolen card",
    "lost card",
    "unauthorized access"
]

medium_priority_keywords = [
    "incorrect charge",
    "billing dispute",
    "payment problem",
    "overdraft fee",
    "interest dispute",
    "delayed refund",
    "unexpected fee",
    "incorrect balance"
]

low_priority_keywords = [
    "general question",
    "information request",
    "account information",
    "how do i",
    "routine service",
    "need information",
    "account inquiry"
]


def assign_priority(complaint_text):

    if not isinstance(complaint_text, str) or not complaint_text.strip():
        return "Medium", "Empty or invalid complaint; default priority applied"

    text = complaint_text.lower()

    # Check high-priority keywords
    for keyword in high_priority_keywords:

        if keyword in text:
            return "High", f"Matched high-priority keyword: {keyword}"

    # Check medium-priority keywords
    for keyword in medium_priority_keywords:

        if keyword in text:
            return "Medium", f"Matched medium-priority keyword: {keyword}"

    # Check low-priority keywords
    for keyword in low_priority_keywords:

        if keyword in text:
            return "Low", f"Matched low-priority keyword: {keyword}"

    # Default priority
    return "Medium", "No matching keyword; default priority applied"


# PART 3: TESTING

if __name__ == "__main__":

    print("\n========== DEPARTMENT ROUTING TEST ==========")

    for product, department in department_mapping.items():

        predicted_department, reason = recommend_department(product)

        print("\nProduct:", product)
        print("Department:", predicted_department)
        print("Result:", reason)

    print("\n========== PRIORITY ASSIGNMENT TEST ==========")

    test_complaints = [
        "I noticed an unauthorized transaction in my account.",
        "Someone has stolen money from my bank account.",
        "I was charged an incorrect charge on my credit card.",
        "I am facing a delayed refund.",
        "I have a general question about my account.",
        "I need information about my account.",
        "I am experiencing an unfamiliar issue."
    ]

    for complaint in test_complaints:

        priority, reason = assign_priority(complaint)

        print("\nComplaint:", complaint)
        print("Priority:", priority)
        print("Reason:", reason)

    print("\n========== INVALID INPUT TEST ==========")

    print("Empty department:",
          recommend_department(""))

    print("Unknown department:",
          recommend_department("Unknown Category"))

    print("Empty priority:",
          assign_priority(""))

    print("\nAll routing tests completed.")
