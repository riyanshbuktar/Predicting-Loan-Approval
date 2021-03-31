import streamlit as st
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt


# Loading the saved Model
model=pickle.load(open('model.pkl','rb'))


def loan_default(features):

    features = np.array(features).astype(np.float64).reshape(1,-1)
    
    predict = model.predict(features)
    probability = model.predict_proba(features)

    return predict, probability


def func_mweacraf():
    
    return 


def main():
    st.title("Loan Approval Prediction")
    dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Lending Clubs Loan Dataset')
)


    classifier_name = st.sidebar.selectbox(
    'Selectclassifier',
    ('Logistic Regression', 'Random Forest')
)

    html_temp = """
    <div style="background-color:#dd88b3 ;padding:10px">
    <h2 style="color:white;text-align:center;">Blood Donation Default Prediction App </h2>
    </div>
    """    
    st.markdown(html_temp, unsafe_allow_html=True)

    Amount_Requested = st.text_input("Amount Requested")
    Risk_Score = st.text_input("Risk Score")
    Debt_to_Income_Ratio= st.text_input("Debt to Income Ratio")
    Employment_Length= st.text_input("Employment Length")



    if st.button("Predict"):
        
        features = [Amount_Requested,Risk_Score,Debt_to_Income_Ratio,Employment_Length]
        predict, proba = loan_default(features)
        if predict[0] == 1:
            

            st.success('Congratulations ! Your loan request has been Approved {} %'.format(round(np.max(proba)*100),2))

        else:
            st.success('Sorry ! but your Loan Request is not approved{} %'.format(round(np.max(proba)*100),2))



#### PLOT DATASET ####
# Project the data onto the 2 primary principal components
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2,
        c=y, alpha=0.8,
        cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

#plt.show()
st.pyplot(fig)



if __name__ == '__main__':
    main()