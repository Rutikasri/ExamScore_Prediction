import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Exam Score Predictor", page_icon="📘", layout="centered")

@st.cache_data
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    return model, scaler, feature_columns

model, scaler, feature_columns = load_artifacts()

st.title("📚 Exam Score Prediction")
st.write("Fill out the student profile and get a predicted exam score.")

with st.form("prediction_form"):
    age = st.number_input("Age", min_value=10, max_value=100, value=20)
    study_hours = st.number_input("Study hours / day", min_value=0.0, max_value=16.0, value=3.0, step=0.1)
    class_attendance = st.number_input("Class attendance (%)", min_value=0.0, max_value=100.0, value=80.0, step=0.1)
    sleep_hours = st.number_input("Sleep hours / night", min_value=0.0, max_value=12.0, value=7.0, step=0.1)

    gender = st.selectbox("Gender", ["male", "female", "other"])
    course = st.selectbox(
        "Course",
        ["bca", "b.sc", "bba", "bcom", "diploma", "mba", "mca", "others"],
    )
    internet_access = st.selectbox("Internet access", ["yes", "no"])
    sleep_quality = st.selectbox("Sleep quality", ["poor", "average", "good"])
    study_method = st.selectbox(
        "Study method",
        ["coaching", "group study", "mixed", "online videos", "self-study"],
    )
    facility_rating = st.selectbox("Facility rating", ["low", "medium", "high"])
    exam_difficulty = st.selectbox("Exam difficulty", ["easy", "moderate", "hard"])

    submit = st.form_submit_button("Predict")

if submit:
    # Build the same feature vector used during training
    row = pd.DataFrame(
        [
            {
                "age": age,
                "study_hours": study_hours,
                "class_attendance": class_attendance,
                "sleep_hours": sleep_hours,
                "gender": gender,
                "course": course,
                "internet_access": internet_access,
                "sleep_quality": sleep_quality,
                "study_method": study_method,
                "facility_rating": facility_rating,
                "exam_difficulty": exam_difficulty,
            }
        ]
    )

    row = pd.get_dummies(row)
    row = row.reindex(columns=feature_columns, fill_value=0)

    # Scale using the same scaler used for training
    row = pd.DataFrame(scaler.transform(row), columns=row.columns)

    prediction = model.predict(row)[0]
    st.success(f"Predicted exam score: **{prediction:.1f}**")

    st.markdown("---")
    st.write(
        "🔎 This model is trained on the provided dataset and gives an estimate based on that data."
    )
