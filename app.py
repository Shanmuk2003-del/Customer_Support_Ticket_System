"""
app.py

Streamlit front-end for the Financial Customer Complaint Classification and
Routing System.

This application reuses the existing backend functions:
  - process_complaint.py -> process_complaint()
  - routing.py           -> recommend_department(), assign_priority()

No model is retrained when the app starts. The saved Logistic Regression
pipeline is loaded once and reused for every prediction.
"""

import streamlit as st

# Reuse the existing backend logic instead of duplicating it.
from process_complaint import process_complaint
from db import (
    init_db,
    insert_complaint,
    get_recent_complaints,
    update_ticket_status,
    get_filtered_complaints,
    get_dashboard_summary,
)


# ---------------------------------------------------------------------------
# Initialize the database once when the app starts.
# st.cache_resource ensures this runs only one time, not on every rerun.
# ---------------------------------------------------------------------------
@st.cache_resource
def _init_database():
    init_db()


_init_database()


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Financial Complaint Classification & Routing",
    page_icon=":bank:",
    layout="centered",
)

st.title("Financial Customer Complaint Classification and Routing System")
st.write(
    "Enter a customer complaint below. The system will classify the complaint "
    "into a product category, route it to the correct department, and assign a priority."
)


# ---------------------------------------------------------------------------
# Dashboard Analytics section (near the top, below the title)
# ---------------------------------------------------------------------------
try:
    summary = get_dashboard_summary()

    # --- A. Key Performance Indicators (metric cards) ---
    st.subheader("Dashboard Analytics")
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    kpi_col1.metric("Total Tickets", summary["total_tickets"])
    kpi_col2.metric("Open Tickets", summary["open_tickets"])
    kpi_col3.metric("In Progress", summary["in_progress_tickets"])
    kpi_col4.metric("Resolved", summary["resolved_tickets"])

    st.divider()

    # --- B. Complaint Category Analysis (bar chart) ---
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.write("**Complaints by Predicted Category**")
        if summary["by_category"]:
            import pandas as pd
            cat_df = pd.DataFrame(summary["by_category"], columns=["Category", "Count"])
            st.bar_chart(cat_df.set_index("Category"))
        else:
            st.info("No tickets to display.")

    # --- C. Department Analysis (bar chart) ---
    with chart_col2:
        st.write("**Complaints by Department**")
        if summary["by_department"]:
            import pandas as pd
            dept_df = pd.DataFrame(summary["by_department"], columns=["Department", "Count"])
            st.bar_chart(dept_df.set_index("Department"))
        else:
            st.info("No tickets to display.")

    # --- D. Priority Analysis (bar chart) ---
    st.write("**Tickets by Priority**")
    if summary["by_priority"]:
        import pandas as pd
        prio_df = pd.DataFrame(summary["by_priority"], columns=["Priority", "Count"])
        st.bar_chart(prio_df.set_index("Priority"))
    else:
        st.info("No tickets to display.")

    # --- E. Ticket Creation Trend (line chart, last 30 days) ---
    st.write("**Daily Ticket Creation Trend (Last 30 Days)**")
    if summary["daily_trend"]:
        import pandas as pd
        trend_df = pd.DataFrame(summary["daily_trend"], columns=["Date", "Count"])
        trend_df["Date"] = pd.to_datetime(trend_df["Date"])
        st.line_chart(trend_df.set_index("Date"))
    else:
        st.info("No tickets created in the last 30 days.")

except Exception as dashboard_error:
    st.warning(
        f"Could not load the dashboard analytics. Error: {dashboard_error}"
    )


# ---------------------------------------------------------------------------
# Input area
# ---------------------------------------------------------------------------
complaint_text = st.text_area(
    "Complaint description",
    placeholder="Example: I noticed an unauthorized transaction on my credit card...",
    height=150,
)

predict_button = st.button("Predict Complaint", type="primary")


# ---------------------------------------------------------------------------
# Priority styling helper
# ---------------------------------------------------------------------------
def priority_badge(priority):
    """Return a colored badge HTML string for a priority level."""
    colors = {
        "High": ("#FFC7C7", "#9C0006", "High"),
        "Medium": ("#FFEB9C", "#9C6500", "Medium"),
        "Low": ("#C6EFCE", "#006100", "Low"),
    }
    bg, fg, label = colors.get(priority, ("#FFFFFF", "#000000", priority))
    return (
        f'<span style="background-color:{bg};color:{fg};'
        f'padding:4px 12px;border-radius:12px;font-weight:bold;'
        f'display:inline-block;">{label} Priority</span>'
    )


# ---------------------------------------------------------------------------
# Handle prediction
# ---------------------------------------------------------------------------
if predict_button:
    if not complaint_text or not complaint_text.strip():
        st.error("Please enter a complaint description before clicking Predict.")
    else:
        try:
            with st.spinner("Processing complaint..."):
                result = process_complaint(complaint_text)

            # --- Save to database (separate step with its own error handling) ---
            try:
                ticket_id = insert_complaint(result)
                st.success(
                    f"Complaint saved successfully. Ticket ID: {ticket_id}"
                )
            except Exception as db_error:
                # Display a clear warning: the prediction succeeded but
                # the record was NOT saved. Never claim it was saved.
                st.warning(
                    f"Complaint was processed, but it could not be saved to the "
                    f"database. Error: {db_error}"
                )

            # --- Category & confidence ---
            st.subheader("Predicted Product Category")
            st.write(result["predicted_category"])
            st.metric(
                "Confidence estimate",
                f"{result['confidence_estimate'] * 100:.1f}%",
            )
            st.caption(
                "Confidence estimates are model scores, not guaranteed probabilities."
            )

            # --- Top categories ---
            with st.expander("See all category scores"):
                for category, score in result["top_categories"]:
                    st.write(f"- **{category}**: {score:.4f}")

            # --- Department ---
            st.subheader("Recommended Department")
            st.info(result["department"])

            # --- Priority ---
            st.subheader("Assigned Priority")
            st.markdown(
                priority_badge(result["priority"]),
                unsafe_allow_html=True,
            )
            st.write(f"**Reason:** {result['priority_reason']}")

        except ValueError as e:
            st.error(f"Input error: {e}")
        except Exception as e:
            st.error(
                f"An unexpected error occurred while processing the complaint: {e}"
            )


# ---------------------------------------------------------------------------
# Ticket History section
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Ticket History")

try:
    tickets = get_recent_complaints(limit=50)

    if not tickets:
        st.info("No tickets have been saved yet. Submit a complaint to create the first ticket.")
    else:
        # Build a clean list of rows for the Streamlit table.
        rows = []
        for t in tickets:
            rows.append(
                {
                    "Ticket ID": t["id"],
                    "Complaint Description": t["complaint_text"],
                    "Predicted Category": t["predicted_category"],
                    "Confidence Estimate": (
                        f"{t['confidence_estimate'] * 100:.1f}%"
                        if t["confidence_estimate"] is not None
                        else "N/A"
                    ),
                    "Department": t["department"],
                    "Priority": t["priority"],
                    "Status": t["status"],
                    "Created At": t["created_at"],
                }
            )

        st.dataframe(rows, width="stretch")

        st.caption(
            f"Showing the {len(rows)} most recent saved tickets. "
            "Priority is assigned using keyword-based business rules, not a trained model."
        )
except Exception as history_error:
    st.warning(
        f"Could not load ticket history. Error: {history_error}"
    )


# ---------------------------------------------------------------------------
# Manage Ticket Status section
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Manage Ticket Status")

try:
    tickets = get_recent_complaints(limit=50)

    if not tickets:
        st.info("No tickets available to update. Submit a complaint first.")
    else:
        # Build option lists for the selectboxes.
        # ticket_options maps the displayed label ("#1 - Credit card") to the ID.
        ticket_options = {
            f"#{t['id']} - {t['predicted_category']} ({t['status']})": t["id"]
            for t in tickets
        }
        status_options = ["Open", "In Progress", "Resolved"]

        col1, col2 = st.columns([2, 1])
        with col1:
            selected_label = st.selectbox(
                "Select a ticket",
                options=list(ticket_options.keys()),
                key="manage_ticket_select",
            )
        with col2:
            new_status = st.selectbox(
                "New status",
                options=status_options,
                key="manage_status_select",
            )

        update_button = st.button("Update Status", type="primary")

        if update_button:
            selected_id = ticket_options[selected_label]
            try:
                success = update_ticket_status(selected_id, new_status)
                if success:
                    st.success(
                        f"Ticket #{selected_id} status updated to '{new_status}'."
                    )
                    # Refresh the ticket history so the new status is visible.
                    st.rerun()
                else:
                    st.error(
                        f"Ticket #{selected_id} was not found. "
                        "It may have been deleted."
                    )
            except ValueError as ve:
                st.error(f"Invalid input: {ve}")
            except Exception as update_error:
                st.error(
                    f"Could not update ticket status. Error: {update_error}"
                )
except Exception as manage_error:
    st.warning(
        f"Could not load the ticket management section. Error: {manage_error}"
    )


# ---------------------------------------------------------------------------
# Search and Filter Tickets section
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Search and Filter Tickets")

# Initialize filter values in session state so they persist across reruns.
if "search_filters" not in st.session_state:
    st.session_state.search_filters = {
        "ticket_id": "",
        "complaint_text": "",
        "predicted_category": "All",
        "department": "All",
        "priority": "All",
        "status": "All",
    }

# Load all tickets once to derive dropdown options (categories, departments, etc.).
all_tickets = get_recent_complaints(limit=1000)

category_options = ["All"] + sorted({t["predicted_category"] for t in all_tickets})
department_options = ["All"] + sorted({t["department"] for t in all_tickets})
priority_options = ["All"] + sorted({t["priority"] for t in all_tickets})
status_options = ["All"] + ["Open", "In Progress", "Resolved"]

with st.form("search_filter_form", clear_on_submit=False):
    col1, col2, col3 = st.columns(3)

    with col1:
        ticket_id_input = st.text_input(
            "Ticket ID",
            value=st.session_state.search_filters["ticket_id"],
            key="search_ticket_id",
            placeholder="e.g. 2",
        )
        complaint_text_input = st.text_input(
            "Complaint description",
            value=st.session_state.search_filters["complaint_text"],
            key="search_complaint_text",
            placeholder="partial text search",
        )

    with col2:
        category_select = st.selectbox(
            "Predicted category",
            options=category_options,
            index=category_options.index(st.session_state.search_filters["predicted_category"])
            if st.session_state.search_filters["predicted_category"] in category_options
            else 0,
            key="search_category",
        )
        department_select = st.selectbox(
            "Department",
            options=department_options,
            index=department_options.index(st.session_state.search_filters["department"])
            if st.session_state.search_filters["department"] in department_options
            else 0,
            key="search_department",
        )

    with col3:
        priority_select = st.selectbox(
            "Priority",
            options=priority_options,
            index=priority_options.index(st.session_state.search_filters["priority"])
            if st.session_state.search_filters["priority"] in priority_options
            else 0,
            key="search_priority",
        )
        status_select = st.selectbox(
            "Status",
            options=status_options,
            index=status_options.index(st.session_state.search_filters["status"])
            if st.session_state.search_filters["status"] in status_options
            else 0,
            key="search_status",
        )

    search_col, clear_col = st.columns([1, 1])
    with search_col:
        apply_button = st.form_submit_button("Search / Apply Filters", type="primary")
    with clear_col:
        clear_button = st.form_submit_button("Clear Filters")

# Handle button actions.
if apply_button or clear_button:
    if clear_button:
        # Restore the unfiltered view.
        st.session_state.search_filters = {
            "ticket_id": "",
            "complaint_text": "",
            "predicted_category": "All",
            "department": "All",
            "priority": "All",
            "status": "All",
        }
    else:
        # Save the current filter values into session state.
        st.session_state.search_filters = {
            "ticket_id": ticket_id_input,
            "complaint_text": complaint_text_input,
            "predicted_category": category_select,
            "department": department_select,
            "priority": priority_select,
            "status": status_select,
        }

# Build the filter arguments, converting "All" to None so the DB function
# ignores those filters.
filters = st.session_state.search_filters
ticket_id_arg = filters["ticket_id"] if filters["ticket_id"].strip() != "" else None
complaint_text_arg = filters["complaint_text"] if filters["complaint_text"].strip() != "" else None
category_arg = filters["predicted_category"] if filters["predicted_category"] != "All" else None
department_arg = filters["department"] if filters["department"] != "All" else None
priority_arg = filters["priority"] if filters["priority"] != "All" else None
status_arg = filters["status"] if filters["status"] != "All" else None

try:
    matched = get_filtered_complaints(
        ticket_id=ticket_id_arg,
        complaint_text=complaint_text_arg,
        predicted_category=category_arg,
        department=department_arg,
        priority=priority_arg,
        status=status_arg,
        limit=100,
    )

    if not matched:
        st.info("No tickets match the current filters. Try clearing some filters.")
    else:
        rows = []
        for t in matched:
            rows.append(
                {
                    "Ticket ID": t["id"],
                    "Complaint Description": t["complaint_text"],
                    "Predicted Category": t["predicted_category"],
                    "Confidence Estimate": (
                        f"{t['confidence_estimate'] * 100:.1f}%"
                        if t["confidence_estimate"] is not None
                        else "N/A"
                    ),
                    "Department": t["department"],
                    "Priority": t["priority"],
                    "Status": t["status"],
                    "Created At": t["created_at"],
                }
            )

        st.dataframe(rows, width="stretch")
        st.caption(
            f"Showing {len(rows)} matching ticket(s). "
            "Priority is assigned using keyword-based business rules, not a trained model."
        )
except Exception as search_error:
    st.warning(
        f"Could not search tickets. Error: {search_error}"
    )


# ---------------------------------------------------------------------------
# Sidebar with instructions
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("How it works")
    st.write(
        "1. Type a complaint description in the text area.\n"
        "2. Click **Predict Complaint**.\n"
        "3. Review the predicted category, department, and priority."
    )
    st.markdown("---")
    st.write("**Priority levels**")
    st.markdown(
        "- **High**: Possible fraud, unauthorized transactions, identity theft.\n"
        "- **Medium**: Billing disputes, fee problems, incorrect charges.\n"
        "- **Low**: General questions, routine service complaints."
    )
    st.markdown("---")
    st.write(
        "Note: Priority is assigned using keyword-based business rules, "
        "not a trained machine learning model."
    )