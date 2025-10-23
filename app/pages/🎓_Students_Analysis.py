# ==========================================================
# Model used :
# RandomForestRegressor → Predicts numeric marks.
# RandomForestClassifier → Predicts categorical outcomes like pass/fail, performance levels, and at-risk status.
# =========================================================

# ==========================================================
# 🎓 ENHANCED STUDENT ANALYTICS DASHBOARD WITH PERFORMANCE & PASS/FAIL PREDICTION
# ==========================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, r2_score
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
faculty_dept = st.session_state.get('department', None)  # Faculty department
st.caption(f"👤 Logged in as: **{username} ({role})**")

# ----------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------
@st.cache_data
def load_data():
    students = get_table("students")
    subjects = get_table("students_subjects")
    return students, subjects

students_df, subjects_df = load_data()


# ==========================================================
# Role-based department access
# ==========================================================
if role == "faculty":
    if not faculty_dept:
        st.error("⚠️ No department assigned to this faculty. Contact admin.")
        st.stop()
    students_df = students_df[students_df["department"] == faculty_dept]
    subjects_df = subjects_df[subjects_df["department"] == faculty_dept] if "department" in subjects_df.columns else subjects_df
    st.info(f"🔒 Faculty access: Viewing data for **{faculty_dept} department only**")

# ----------------------------------------------------------
# SIDEBAR FILTERS
# ----------------------------------------------------------
st.sidebar.header("Filters")
dept_filter = st.sidebar.selectbox("Select Department", ["All"] + sorted(students_df["department"].unique()))
year_filter = st.sidebar.selectbox("Select Year", ["All"] + sorted(students_df["year"].unique().astype(str)))

filtered_students = students_df.copy()
if dept_filter != "All":
    filtered_students = filtered_students[filtered_students["department"] == dept_filter]
if year_filter != "All":
    filtered_students = filtered_students[filtered_students["year"] == int(year_filter)]

# PAGE TITLE
st.title("🎓 Enhanced Student Analytics & Prediction Dashboard")
# ----------------------------------------------------------
# OVERVIEW
# ----------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    fig = px.histogram(filtered_students, x="avg_grade", nbins=20,
                       hover_data=["student_id"], title="Distribution of Average Grades")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.pie(filtered_students, names="department", title="Department Distribution")
    st.plotly_chart(fig, use_container_width=True)

heat_df = filtered_students.pivot_table(index="department", columns="year", values="attendance_pct", aggfunc="mean")
fig, ax = plt.subplots(figsize=(8,4))
sns.heatmap(heat_df, annot=True, cmap="YlGnBu", fmt=".1f", ax=ax)
ax.set_title("Average Attendance % by Department & Year")
st.pyplot(fig)

# ----------------------------------------------------------
# STUDENT PERFORMANCE LABEL
# ----------------------------------------------------------
if "library_visits_per_month" not in filtered_students.columns:
    filtered_students["library_visits_per_month"] = np.round(np.random.uniform(0,10,len(filtered_students)),0)

def performance_category(row):
    if row['avg_grade'] >= 8 and row['attendance_pct'] >= 85:
        return 'High'
    elif row['avg_grade'] >= 6:
        return 'Medium'
    else:
        return 'Low'

filtered_students['performance'] = filtered_students.apply(performance_category, axis=1)

# ----------------------------------------------------------
# PERFORMANCE DISTRIBUTION
# ----------------------------------------------------------
st.subheader("📊 Student Performance Distribution")
fig = px.pie(filtered_students, names="performance", title="Overall Performance Distribution")
st.plotly_chart(fig, use_container_width=True)

st.subheader("🏫 Department-wise Performance")
dept_perf = filtered_students.groupby(["department","performance"]).size().reset_index(name='count')
fig = px.bar(dept_perf, x="department", y="count", color="performance", barmode="group",
             text='count', title="Performance by Department")
st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# ATTENDANCE CATEGORY BASED ON THRESHOLD
# ----------------------------------------------------------
st.subheader("📊 Attendance Category (Threshold: 75%)")
col1, col2 = st.columns(2)
attendance_threshold = 75
filtered_students["attendance_category"] = np.where(filtered_students["attendance_pct"] < attendance_threshold,
                                                    "Poor Attendance", "Good Attendance")

fig = px.pie(filtered_students, names="attendance_category", title=f"Attendance Category (Threshold: {attendance_threshold}%)",
             color="attendance_category", color_discrete_map={"Poor Attendance":"red","Good Attendance":"green"})
with col1:
    st.plotly_chart(fig, use_container_width=True)

dept_attendance = filtered_students.groupby(["department","attendance_category"]).size().reset_index(name="count")
fig = px.bar(dept_attendance, x="department", y="count", color="attendance_category", barmode="group",
             title="Department-wise Attendance Category Distribution",
             color_discrete_map={"Poor Attendance":"red","Good Attendance":"green"})
with col2:
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# ENGAGEMENT & PERFORMANCE CHARTS
# ----------------------------------------------------------
st.subheader("📈 Engagement & Performance")
fig = px.scatter(filtered_students, x="attendance_pct", y="avg_grade",
                 color="attendance_trend", size="avg_study_hours",
                 hover_data=["student_id", "department", "year", "performance"],
                 title="Attendance vs Average Grade")
st.plotly_chart(fig, use_container_width=True)

fig = px.scatter(filtered_students, x="avg_study_hours", y="extracurricular_score",
                 color="department", size="avg_grade",
                 hover_data=["student_id", "year", "performance"],
                 title="Study Hours vs Extracurricular Score")
st.plotly_chart(fig, use_container_width=True)

top_students = filtered_students.sort_values("avg_grade", ascending=False).head(10)
fig = px.bar(top_students, x="student_id", y="avg_grade", color="department",
             hover_data=["year", "performance"], title="Top 10 Students by Average Grade")
st.plotly_chart(fig, use_container_width=True)

trend_counts = filtered_students["attendance_trend"].value_counts().reset_index()
trend_counts.columns = ["trend","count"]
fig = px.pie(trend_counts, names="trend", values="count", title="Attendance Trend Distribution")
st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# SUBJECT PERFORMANCE CHARTS
# ----------------------------------------------------------
st.subheader("📚 Subject Performance Analysis")
avg_subject_marks = subjects_df.groupby("subject")[["month1_marks","month2_marks","month3_marks","model_marks","practicals_marks"]].mean(numeric_only=True).reset_index()
avg_subject_marks["Overall_Avg"] = avg_subject_marks[["month1_marks","month2_marks","month3_marks","model_marks","practicals_marks"]].mean(axis=1)
fig = px.bar(avg_subject_marks, x="subject", y="Overall_Avg", title="Average Marks per Subject")
st.plotly_chart(fig, use_container_width=True)

subjects_melted = subjects_df.melt(id_vars=["subject"], value_vars=["month1_marks","month2_marks","month3_marks","model_marks","practicals_marks"],
                                   var_name="Exam", value_name="Marks")
fig = px.box(subjects_melted, x="subject", y="Marks", color="Exam", title="Subject-wise Marks Distribution")
st.plotly_chart(fig, use_container_width=True)

avg_exam_marks = subjects_df[["month1_marks","month2_marks","month3_marks","model_marks","practicals_marks"]].mean()
fig = px.line(x=avg_exam_marks.index, y=avg_exam_marks.values, markers=True,
              title="Average Marks Trend Across Exams", labels={"x":"Exam","y":"Average Marks"})
st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# SEMESTER MARKS PREDICTION (STUDENT LEVEL)
# ----------------------------------------------------------
subjects_df["semester_marks"] = (
    0.15*subjects_df["month1_marks"] + 0.15*subjects_df["month2_marks"] + 0.15*subjects_df["month3_marks"] +
    0.25*subjects_df["model_marks"] + 0.30*subjects_df["practicals_marks"] +
    np.random.normal(0,2,len(subjects_df))
).clip(0,100)

marks_agg = subjects_df.groupby("student_id").mean(numeric_only=True).reset_index()
merged = pd.merge(marks_agg, students_df, on="student_id", how="left")  # merged exists now

X_reg = merged[["month1_marks","month2_marks","month3_marks","model_marks","practicals_marks","attendance_pct","avg_study_hours"]]
y_reg = merged["semester_marks"]
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
model_reg = RandomForestRegressor(n_estimators=120, random_state=42)
model_reg.fit(Xr_train, yr_train)
preds = model_reg.predict(Xr_test)

st.subheader("📈 Semester Marks Prediction")
st.text(f"R² Score: {r2_score(yr_test,preds):.2f}, MAE: {mean_absolute_error(yr_test,preds):.2f}")
fig = px.scatter(x=yr_test, y=preds,
                 labels={'x':'Actual Marks','y':'Predicted Marks'},
                 hover_data={"Actual": yr_test, "Predicted": preds},
                 title="Actual vs Predicted Semester Marks")
st.plotly_chart(fig, use_container_width=True)

importances = pd.Series(model_reg.feature_importances_, index=X_reg.columns).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(6,4))
importances.plot(kind='barh', color='lightgreen', ax=ax)
ax.set_title("Feature Importance (Semester Marks Prediction)")
st.pyplot(fig)

# ----------------------------------------------------------
# PREDICT SEMESTER MARKS PER SUBJECT
# ----------------------------------------------------------
st.subheader("📈 Predicted Semester Marks per Subject")
subjects_df = subjects_df.merge(students_df[["student_id","attendance_pct","avg_study_hours"]], on="student_id", how="left")

pred_marks_list = []
for subj in subjects_df["subject"].unique():
    subj_df = subjects_df[subjects_df["subject"]==subj].copy()
    X_subj = subj_df[["month1_marks","month2_marks","month3_marks","model_marks","practicals_marks","attendance_pct","avg_study_hours"]]
    y_subj = 0.15*subj_df["month1_marks"] + 0.15*subj_df["month2_marks"] + 0.15*subj_df["month3_marks"] + \
             0.25*subj_df["model_marks"] + 0.30*subj_df["practicals_marks"]
    model_subj = RandomForestRegressor(n_estimators=120, random_state=42)
    model_subj.fit(X_subj, y_subj)
    subj_df["predicted_sem_marks"] = model_subj.predict(X_subj)
    pred_marks_list.append(subj_df[["student_id","subject","predicted_sem_marks"]])

pred_marks_df = pd.concat(pred_marks_list)
st.dataframe(pred_marks_df.head(20))

# ----------------------------------------------------------
# INDIVIDUAL STUDENT MARKS TREND WITH PREDICTED SEMESTER MARK
# ----------------------------------------------------------
st.subheader("📈 Individual Student Marks Trend (Per Subject + Predicted Semester Marks)")
sample_student_id = st.selectbox("Select Student ID for Subject-level Prediction", subjects_df["student_id"].unique())

student_subjects = pred_marks_df[pred_marks_df["student_id"]==sample_student_id].merge(
    subjects_df[["student_id","subject","month1_marks","month2_marks","month3_marks","model_marks","practicals_marks"]],
    on=["student_id","subject"]
)

# Melt exam marks
plot_df = student_subjects.melt(
    id_vars=["subject","predicted_sem_marks"], 
    value_vars=["month1_marks","month2_marks","month3_marks","model_marks","practicals_marks"],
    var_name="Exam", value_name="Marks"
)

# Add predicted semester marks as a separate row for each subject
pred_sem_df = student_subjects[["subject","predicted_sem_marks"]].copy()
pred_sem_df = pred_sem_df.rename(columns={"predicted_sem_marks":"Marks"})
pred_sem_df["Exam"] = "Predicted Semester"

plot_df = pd.concat([plot_df, pred_sem_df], ignore_index=True)

fig = px.line(plot_df, x="Exam", y="Marks", color="subject", markers=True,
              title=f"Student ID {sample_student_id} Marks Trend with Predicted Semester Marks")
st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# SUBJECT-LEVEL PASS/FAIL PREDICTION
# ----------------------------------------------------------
# Merge predicted semester marks into subjects_df
subjects_df = subjects_df.merge(pred_marks_df, on=["student_id","subject"], how="left")

# Now create subject-level pass/fail
st.subheader("🎯 Subject-level Pass/Fail Prediction (Threshold: 50 Marks + Attendance >=75%)")
subjects_df["pass_fail_subject"] = np.where(
    (subjects_df["predicted_sem_marks"] >=50) & (subjects_df["attendance_pct"]>=75),
    "Pass","Fail"
)
st.dataframe(subjects_df[["student_id","subject","attendance_pct","predicted_sem_marks","pass_fail_subject"]])



# The rest of your original code (At-Risk Analysis, Performance Prediction, Pass/Fail Student-level) remains unchanged

# ----------------------------------------------------------
# AT-RISK STUDENTS ANALYSIS
# ----------------------------------------------------------
students_df["at_risk_flag"] = np.where((students_df["attendance_pct"]<65)|(students_df["avg_grade"]<5.5),1,0)
st.subheader("🚨 At-Risk Students Analysis")
fig = px.pie(students_df, names="at_risk_flag", title="At-Risk vs Safe Students")
st.plotly_chart(fig, use_container_width=True)

at_risk_counts = students_df.groupby("department")["at_risk_flag"].sum().reset_index()
fig = px.bar(at_risk_counts, x="department", y="at_risk_flag", title="At-Risk Students by Department")
st.plotly_chart(fig, use_container_width=True)

st.subheader("🤖 Predict At-Risk Students")
X = students_df[["attendance_pct","avg_grade","assignments_submitted","extracurricular_score","avg_study_hours"]]
y = students_df["at_risk_flag"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=120, random_state=42)
model.fit(X_train, y_train)
pred_class = model.predict(X_test)
st.text("Classification Report:")
st.code(classification_report(y_test, pred_class), language='text')
conf = confusion_matrix(y_test, pred_class)
st.write(pd.DataFrame(conf, columns=["Pred 0","Pred 1"], index=["True 0","True 1"]))

# ----------------------------------------------------------
# STUDENT PERFORMANCE PREDICTION
# ----------------------------------------------------------
st.subheader("📈 Predict Overall Student Performance")
perf_features = ["attendance_pct","avg_grade","assignments_submitted","extracurricular_score","avg_study_hours","library_visits_per_month"]
X_perf = filtered_students[perf_features]
y_perf = filtered_students["performance"]
X_train, X_test, y_train, y_test = train_test_split(X_perf, y_perf, test_size=0.2, random_state=42)
perf_model = RandomForestClassifier(n_estimators=120, random_state=42)
perf_model.fit(X_train, y_train)
y_pred_perf = perf_model.predict(X_test)

st.text("Classification Report for Performance Prediction:")
st.code(classification_report(y_test, y_pred_perf), language='text')
conf_perf = confusion_matrix(y_test, y_pred_perf)
st.write(pd.DataFrame(conf_perf, columns=perf_model.classes_, index=perf_model.classes_))

importances_perf = pd.Series(perf_model.feature_importances_, index=perf_features).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(6,4))
importances_perf.plot(kind='barh', color='orange', ax=ax)
ax.set_title("Feature Importance (Performance Prediction)")
st.pyplot(fig)

# ----------------------------------------------------------
# PASS/FAIL PREDICTION (ATTENDANCE <75% FAIL)
# ----------------------------------------------------------
st.subheader("🎯 Pass/Fail Prediction (Attendance <75% = Fail)")
merged["pass_fail_adjusted"] = np.where(merged["attendance_pct"] < 75, "Fail",
                                        np.where(merged["semester_marks"] >= 50, "Pass", "Fail"))

pass_features = ["attendance_pct","avg_grade","assignments_submitted","extracurricular_score","avg_study_hours","library_visits_per_month"]
X_pass = merged[pass_features]
y_pass = merged["pass_fail_adjusted"]

X_train, X_test, y_train, y_test = train_test_split(X_pass, y_pass, test_size=0.2, random_state=42)
pass_model = RandomForestClassifier(n_estimators=120, random_state=42)
pass_model.fit(X_train, y_train)
y_pred_pass = pass_model.predict(X_test)

st.text("Classification Report for Pass/Fail Prediction:")
st.code(classification_report(y_test, y_pred_pass), language='text')

conf_pass = confusion_matrix(y_test, y_pred_pass)
st.write("Confusion Matrix (Pass/Fail):")
st.write(pd.DataFrame(conf_pass, columns=pass_model.classes_, index=pass_model.classes_))

st.subheader("📊 Feature Importance (Pass/Fail Prediction)")
importances_pass = pd.Series(pass_model.feature_importances_, index=pass_features).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(6,4))
importances_pass.plot(kind='barh', color='purple', ax=ax)
ax.set_title("Feature Importance (Pass/Fail Prediction)")
st.pyplot(fig)

# ----------------------------------------------------------
# ADDITIONAL INSIGHT CHARTS
# ----------------------------------------------------------
st.subheader("📌 Additional Insight Charts")
fig = px.box(filtered_students, x="performance", y="attendance_pct",
             color="performance", title="Attendance % by Performance Category")
st.plotly_chart(fig, use_container_width=True)

fig = px.histogram(filtered_students, x="avg_study_hours", nbins=20,
                   color="performance", title="Distribution of Average Study Hours by Performance")
st.plotly_chart(fig, use_container_width=True)

fig = px.scatter(filtered_students, x="library_visits_per_month", y="avg_grade",
                 color="performance", size="avg_study_hours",
                 hover_data=["student_id","department","year"],
                 title="Library Visits vs Average Grade")
st.plotly_chart(fig, use_container_width=True)

merged["attendance_category"] = np.where(merged["attendance_pct"] < 75, "Poor Attendance", "Good Attendance")
fig = px.scatter(merged, x="attendance_pct", y="semester_marks",
                 color="attendance_category", hover_data=["student_id","department"],
                 title="Attendance % vs Predicted Semester Marks")
st.plotly_chart(fig, use_container_width=True)

feature_cols = ["attendance_pct","avg_grade","assignments_submitted","extracurricular_score",
                "avg_study_hours","library_visits_per_month","semester_marks"]
corr_df = merged[feature_cols].corr()
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(corr_df, annot=True, cmap="coolwarm", ax=ax)
ax.set_title("Feature Correlation Heatmap")
st.pyplot(fig)

pass_counts = merged["pass_fail_adjusted"].value_counts().reset_index()
pass_counts.columns = ["Status", "Count"]
fig = px.pie(pass_counts, names="Status", values="Count", title="Pass vs Fail Students (Adjusted for Attendance)",
             color="Status", color_discrete_map={"Pass":"green","Fail":"red"})
st.plotly_chart(fig, use_container_width=True)
