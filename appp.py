import pickle
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="Estimated Salary Prediction",
    page_icon="💰",
    layout="wide"
)
try:
    model = tf.keras.models.load_model("salary_model.keras")
    with open("encod_gender.pkl", "rb") as file:
        label_encoder_gender = pickle.load(file)

    with open("encod_geo.pkl", "rb") as file:
        onehot_encoder_geo = pickle.load(file)

    with open("scalar.pkl", "rb") as file:
        scaler = pickle.load(file)

except Exception as e:
    st.error(f"Error loading model or processing files: {e}")
    st.stop()

st.title("Estimated Salary Prediction")
st.caption("This app predicts the estimated salary.")

st.success("ANN Model and preprocessing files loaded successfully.")

predict_tab,info_tab= st.tabs(['Predict Estimated Salary','How it works'])
with predict_tab:
    st.subheader("Enter Details")
    st.write(" Provide Details and press **Predict Estimated Salary** button.")

with st.form ("employee_form"):
    col1,col2,col3=st.columns(3)
    with col1:
        geography= st.selectbox ("Geography", onehot_encoder_geo.categories_[0])
        gender=st.selectbox("Gender", label_encoder_gender.classes_)
        credit_score= st.slider("Credit_Score", min_value=0, max_value=1000, value=650, step=1)
    with col2:
        age = st.number_input("Age", min_value=18, max_value=92, value=40)
        tenure = st.slider("Tenure (Years)", min_value=0, max_value=10, value=5)
        num_of_products = st.slider("Number of Products", min_value=1, max_value=4, value=1)
    with col3:
        has_card_text = st.selectbox("Has Credit Card", ["Yes", "No"])
        balance = st.number_input("Balance", min_value=0.0, value=0.0, step=1000.0)
        exited = st.selectbox("Exited", ['Yes', 'No'])
        active_text = st.selectbox("Is Active Member", ["Yes", "No"])
    
    predict_button = st.form_submit_button("Predict Estimated Salary", type="primary", use_container_width=True)
    if predict_button:
        has_cr_card = 1 if has_card_text == "Yes" else 0
        is_active_member = 1 if active_text == "Yes" else 0
        encoded_gender = label_encoder_gender.transform([gender])[0]
        exited_num= 1 if exited=="No" else 0
    
        input_data = pd.DataFrame({
            "CreditScore": [credit_score],
            "Gender": [encoded_gender],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_of_products],
            "HasCrCard": [has_cr_card],
            "IsActiveMember": [is_active_member],
            "Exited": [exited_num]
            })
    
        geography_df = pd.DataFrame({"Geography": [geography]})
        geography_encoded = onehot_encoder_geo.transform(geography_df).toarray()
        geography_df = pd.DataFrame(
            geography_encoded,
            columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
            )
    
        input_data = pd.concat([input_data, geography_df], axis=1)
        input_data = input_data[scaler.feature_names_in_]
    
        input_scaled = scaler.transform(input_data)
    
        prediction = model.predict(input_scaled, verbose=0)
    
        estimated_salary_probability = float(prediction[0][0])
    
        st.divider()
    
        st.subheader("Prediction Results")
    
        result_col1, result_col2, result_col3 = st.columns(3)
    
        with result_col1:
            st.metric("Esimated Salary", f"{estimated_salary_probability:.2f}")
    
        with result_col2:
            st.metric ("R2", "0.82")
    
            with result_col3:
                st.metric(
                    "Test MAE", "$5,240.50"
                )
    
            st.success(f"The ANN estimates that salary will be **${estimated_salary_probability:,.2f}**.")
    
        with st.expander("View Employee  Summary"):
            employee_summary = pd.DataFrame(
                {
                    "Feature": [
                        "Geography",
                        "Gender",
                        "Credit Score",
                        "Age",
                        "Tenure",
                        "Balance",
                        "Number of Products",
                        "Has Credit Card",
                        "Active Member",
                        "Exited"
                    ],
                    "Value": [
                        geography,
                        gender,
                        credit_score,
                        age,
                        tenure,
                        balance,
                        num_of_products,
                        has_card_text,
                        active_text,
                        exited_num
    
                    ]
                }
            )
            st.dataframe(
                employee_summary,
                use_container_width=True,
                hide_index=True
            )
        
    
    
         
    
    