import streamlit as st
import pickle
import numpy as np
import os

@st.cache(allow_output_mutation=True)
def load_model():
    model_path = os.path.join("models", "titanic_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def main():
    st.title("Titanic Survival Prediction")
    st.write("Enter passenger details:")

    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
    sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
    fare = st.number_input("Fare", min_value=0.0, max_value=500.0, value=32.0)
    embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

    if st.button("Predict"):
        model = load_model()
        # Map categorical variables to numeric values.
        sex_numeric = 1 if sex.lower() == "male" else 0
        embarked_mapping = {"C": 0, "Q": 1, "S": 2}
        embarked_numeric = embarked_mapping[embarked]
        features = np.array([[pclass, sex_numeric, age, sibsp, parch, fare, embarked_numeric]])
        prediction = model.predict(features)[0]
        result = "Survived" if prediction == 1 else "Did not survive"
        st.write(f"Prediction: {result}")

if __name__ == "__main__":
    main()
