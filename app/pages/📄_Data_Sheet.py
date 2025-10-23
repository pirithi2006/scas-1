# ==========================================================
# 📄 data_manager.py — Secure Data Sheet & Upload Manager
# ==========================================================
import streamlit as st
import pandas as pd
import sqlite3
from db import get_table

# ==========================================================
# ✅ Login & Role Check
# ==========================================================
if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    st.error("⚠️ You must be logged in to access this page.")
    st.page_link("👤_Login.py", label="🔑 Go to Login Page")
    st.stop()

role = st.session_state.get('role', 'guest')
username = st.session_state.get('username', 'Unknown')

st.title("📄 Data Sheet & Upload Data")
st.caption(f"👤 Logged in as: **{username} ({role})**")

# ==========================================================
# 🔒 Role-Based Dataset Access
# ==========================================================
if role == "admin":
    dataset_options = ["students", "students_subjects", "faculty", "energy", "canteen", "facility_logs"]
else:
    # Faculty users can only access student-related tables
    dataset_options = ["students", "students_subjects"]

selected_dataset = st.selectbox("Select Table / Dataset:", dataset_options)

# ==========================================================
# 1️⃣ Upload CSV to Database (Admin + Faculty Rules)
# ==========================================================
st.subheader("⬆️ Upload CSV to Table")

uploaded_file = st.file_uploader("Choose a CSV file to upload", type="csv")

# Define upload permissions
can_upload = (role == "admin") or (role == "faculty" and selected_dataset in ["students", "students_subjects"])

if uploaded_file is not None:
    try:
        upload_df = pd.read_csv(uploaded_file)

        # Connect to DB
        conn = sqlite3.connect("SCAPS.db")
        c = conn.cursor()

        # Check table columns
        c.execute(f"PRAGMA table_info({selected_dataset})")
        table_columns = [info[1] for info in c.fetchall()]
        missing_cols = [col for col in table_columns if col not in upload_df.columns]

        if missing_cols:
            st.error(f"The uploaded CSV is missing these columns: {missing_cols}")
        else:
            if can_upload:
                if st.button("Upload to Database"):
                    upload_df.to_sql(selected_dataset, conn, if_exists="replace", index=False)
                    st.success(f"✅ Data uploaded successfully to table '{selected_dataset}'!")
            else:
                st.warning("🔒 You don't have permission to upload this dataset.")

        conn.close()
    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    if role != "admin":
        st.info("ℹ️ Faculty users can upload only **students** or **students_subjects** data.")

# ==========================================================
# 2️⃣ Load Current Dataset
# ==========================================================
try:
    df = get_table(selected_dataset)
except Exception as e:
    st.error(f"⚠️ Unable to load table '{selected_dataset}': {e}")
    st.stop()

# ==========================================================
# 3️⃣ Filters (Categorical Columns Only)
# ==========================================================
st.subheader("🔎 Filters")

categorical_cols = df.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()
col_count = 3
cols = st.columns(col_count)
filter_values = {}

for i, col in enumerate(categorical_cols):
    with cols[i % col_count]:
        options = ["All"] + df[col].dropna().unique().tolist()
        selected = st.multiselect(f"{col}:", options, default=["All"])
        if "All" in selected:
            selected = df[col].dropna().unique().tolist()
        filter_values[col] = selected

# Apply filters
for col, val in filter_values.items():
    df = df[df[col].isin(val)]

# ==========================================================
# 4️⃣ Display Filtered Data
# ==========================================================
st.subheader(f"📊 Filtered Data — '{selected_dataset}'")
st.dataframe(df, use_container_width=True)

# ==========================================================
# 5️⃣ Download Button
# ==========================================================
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download CSV",
    data=csv,
    file_name=f"{selected_dataset}_filtered.csv",
    mime="text/csv"
)
