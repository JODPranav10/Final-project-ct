import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import streamlit as st

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("student.csv")

# -----------------------------
# Rename columns
# -----------------------------
df.rename(columns={
    'G3': 'marks',
    'absences': 'attendance',
    'studytime': 'assignments'
}, inplace=True)

# -----------------------------
# Convert attendance (absences → attendance)
# -----------------------------
df['attendance'] = 100 - df['attendance']

# -----------------------------
# Create Risk Label
# -----------------------------
def risk(row):
    if row['marks'] < 10 and row['attendance'] < 60:
        return 2
    elif row['marks'] < 12 or row['attendance'] < 75:
        return 1
    else:
        return 0

df['risk'] = df.apply(risk, axis=1)

# -----------------------------
# Train Model
# -----------------------------
X = df[['marks', 'attendance', 'assignments']]
y = df['risk']

model = DecisionTreeClassifier()
model.fit(X, y)

# -----------------------------
# UI STARTS HERE
# -----------------------------
st.title("🎓 AI Student Dropout Risk Predictor")

st.subheader("Enter Student Details")

marks = st.number_input("Marks (0–20)", 0, 20)
attendance = st.number_input("Attendance (%)", 0, 100)
assignments = st.number_input("Study Time (1–4)", 1, 4)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Risk"):

    result = model.predict([[marks, attendance, assignments]])[0]

    # Risk Label
    if result == 2:
        st.error("⚠️ High Dropout Risk")
    elif result == 1:
        st.warning("⚠️ Medium Risk")
    else:
        st.success("✅ Low Dropout Risk")

    # -----------------------------
    # Risk Score
    # -----------------------------
    risk_score = int((100 - attendance)*0.4 + (20 - marks)*3)

    st.subheader("📊 Risk Score")
    st.write(f"Risk Score: {risk_score}%")

    # -----------------------------
    # Explanation
    # -----------------------------
    st.subheader("📊 Explanation")

    if attendance < 60:
        st.write("- Low attendance is a major factor")
    if marks < 10:
        st.write("- Very low marks detected")
    if assignments <= 2:
        st.write("- Low study time affects performance")

    # -----------------------------
    # Suggestions
    # -----------------------------
    st.subheader("💡 Suggestions")

    if attendance < 75:
        st.write("✔ Improve attendance to at least 75%")
    if marks < 12:
        st.write("✔ Focus on improving marks")
    if assignments <= 2:
        st.write("✔ Increase study time")

    # -----------------------------
    # Comparison with average
    # -----------------------------
    st.subheader("📈 Comparison with Average")

    avg_marks = df['marks'].mean()
    avg_attendance = df['attendance'].mean()

    if marks < avg_marks:
        st.write("⬇ Marks below average")
    else:
        st.write("⬆ Marks above average")

    if attendance < avg_attendance:
        st.write("⬇ Attendance below average")
    else:
        st.write("⬆ Attendance above average")

# -----------------------------
# Dataset Dashboard
# -----------------------------
st.subheader("📊 Dataset Insights")

st.write("Marks Distribution")
st.bar_chart(df['marks'])

st.write("Attendance Distribution")
st.bar_chart(df['attendance'])

st.write("Risk Distribution")
st.bar_chart(df['risk'].value_counts())
st.subheader("🧠 Key Risk Factor")

factors = {
    "Marks": marks,
    "Attendance": attendance,
    "Study Time": assignments
}

lowest = min(factors, key=factors.get)

st.write(f"Main issue: **{lowest}** is lowest")