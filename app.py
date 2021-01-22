import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report,accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier   
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import pickle
import shap

# Importing the data set 
from sklearn.datasets import load_boston
boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target

# Saving the model 
X = boston.drop(['ZN', 'INDUS', 'CHAS','PTRATIO','B','MEDV'], axis = 1)
Y = boston['MEDV']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.24, random_state=101)
model = RandomForestRegressor()
model.fit(X, Y)
saved_model=pickle.dumps(model)

# Setting of Background 
page_bg_img = '''
<style>
body {
background-image: url("https://img.freepik.com/free-vector/bright-background-with-dots_1055-3132.jpg?size=338&ext=jpg&ga=GA1.2.1846610357.1604275200");
background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)


st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
# Boston House Price Prediction App
This app predicts the **Boston House Price**!
""")
st.sidebar.title("Prediction App")

nav=st.sidebar.radio("",["Home","Data Visualisation","Prediction"])


if nav=="Home":
    st.write("## Description of Predictor App")
    st.write("### The app predcits the median value of owner occupied homes in $1000s")

    # st.write("## Dataset Description")

    st.write(boston_dataset.DESCR)

    if st.checkbox("Show Tabulated"):
        st.table(boston)

if nav=="Data Visualisation":
    st.sidebar.write("# Choose From the following")
    st.header("Visualisation")
    st.write("### Data visualization is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data ")
    if st.sidebar.checkbox("Dist Plot"):
        st.write("## Dist Plot")
        sns.set(rc={'figure.figsize':(11.7,8.27)})
        sns.distplot(boston['MEDV'], bins=30)
        st.pyplot()

    if st.sidebar.checkbox("Correlation Matrix"):
        st.write("## Correlation Matrix")
        correlation_matrix = boston.corr().round(2)
        sns.heatmap(data=correlation_matrix, annot=True)
        st.pyplot()


    if st.sidebar.checkbox("Histogram"):
        st.write("## Histogram")
        boston.hist(edgecolor='black',figsize=(10,8))
        st.pyplot()
    # if st.sidebar.checkbox("Pair Plot"):
    #     st.write("## Pair  Plot")
    #     sns.pairplot(boston,hue="MEDV")
    #     st.pyplot()
    if st.sidebar.checkbox("Count Plot"):
        st.write("## Count Plot")
        sns.countplot(x="MEDV",data=boston)
        st.pyplot()
    # if st.sidebar.checkbox("6"):
    #     st.write("## 6")
if nav=="Prediction":
    st.write("## Prediction of MEDV")
    st.sidebar.header("Specify Input Parameters")


    val1=st.sidebar.slider("CRIM",float(boston.CRIM.min()),float(boston.CRIM.max()),float(boston.CRIM.mean()))
    val2=st.sidebar.slider("NOX",float(boston.NOX.min()),float(boston.NOX.max()),float(boston.NOX.mean()))
    val3=st.sidebar.slider("RM",float(boston.RM.min()),float(boston.RM.max()),float(boston.RM.mean()))
    val4=st.sidebar.slider("AGE",float(boston.AGE.min()),float(boston.AGE.max()),float(boston.AGE.mean()))
    val5=st.sidebar.slider("DIS",float(boston.DIS.min()),float(boston.DIS.max()),float(boston.DIS.mean()))    
    val6=st.sidebar.slider("RAD",float(boston.RAD.min()),float(boston.RAD.max()),float(boston.RAD.mean()))
    val7=st.sidebar.slider("TAX",float(boston.TAX.min()),float(boston.TAX.max()),float(boston.TAX.mean()))    
    val8=st.sidebar.slider("LSTAT",float(boston.LSTAT.min()),float(boston.LSTAT.max()),float(boston.LSTAT.mean()))    
    val=[[val1,val2,val3,val4,val5,val6,val7,val8]]
    model_from_pickle=pickle.loads(saved_model)
    prediction=model_from_pickle.predict(val)

    if st.button("Predict"):
        st.success(f"Rate is {(prediction)}")
    
# Explaining the model's predictions using SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    st.header('Feature Importance')
    plt.title('Feature importance based on SHAP values')
    shap.summary_plot(shap_values, X)
    st.pyplot(bbox_inches='tight')

    plt.title('Feature importance based on SHAP values (Bar)')
    shap.summary_plot(shap_values, X, plot_type="bar")
    st.pyplot(bbox_inches='tight')









