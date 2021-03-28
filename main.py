import streamlit as st
import pickle
import numpy as np

# Loading the saved Model
model=pickle.load(open('model.pkl','rb'))


def loan_default(features):

    features = np.array(features).astype(np.float64).reshape(1,-1)
    
    predict = model.predict(features)
    probability = model.predict_proba(features)

    return predict, probability


def main():
    st.title("Loan Approval Prediction")
    html_temp = """
    <div style="background-color:#dd88b3 ;padding:10px">
    <h2 style="color:white;text-align:center;">Blood Donation Default Prediction App </h2>
    </div>
    """    
    st.markdown(html_temp, unsafe_allow_html=True)

    Amount Requested = st.text_input("Amount Requested")
    Risk_Score = st.text_input("Frequency(times)")
    Debt to Income Ratio= st.text_input("Monetory(cc blood)")
    Employment Length= st.text_input("Time(months) ")



    if st.button("Predict"):
        
        features = [Amount Requested,Risk_Score,Debt to Income Ratio,Employment Length]
        predict, proba = loan_default(features)
        if predict[0] == 1:
            

            st.success('Congratulations ! Your loan request has been Approved {} %'.format(round(np.max(proba)*100),2))

        else:
            st.success('Sorry but you Loan Request is not approved{} %'.format(round(np.max(proba)*100),2))




if __name__ == '__main__':
    main()