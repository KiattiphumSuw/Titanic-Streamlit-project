import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import tree

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def main():

    model = None
    cm_dp = None
    rou_dp = None

    st.set_option('deprecation.showPyplotGlobalUse', False)

    @st.cache_data
    def load_data():
        select_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked', 'Survived']
        data = pd.read_csv('titanic.csv')
        data = data[select_features]
        return data
    
    df = load_data()

    @st.cache_data
    def split(df):
        y = df['Survived']
        x = df.drop(columns=['Survived'])
        return train_test_split(x, y, test_size=0.3, random_state=0)
    
    X_train, X_test, y_train, y_test = split(df)

    @st.cache_data
    def clean_data_process(X_train):
        categorical_cols = [cname for cname in X_train.columns if X_train[cname].nunique() < 10 and 
                                                                  X_train[cname].dtype == "object"]

        numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]

        my_cols = categorical_cols + numerical_cols
        X_train = X_train[my_cols].copy()

        numerical_transformer = SimpleImputer(strategy='constant')

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
        ])
        return preprocessor
    
    preprocessor = clean_data_process(X_train)

    def plot_metrics(y_pred, y_test):

        cm = confusion_matrix(y_test, y_pred)
        cm_dp = ConfusionMatrixDisplay(confusion_matrix=cm)

        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        rou_dp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        return cm_dp, rou_dp
    
   

    st.title("Binary Classification Web App")
    st.markdown("""This web app use data from kaggle competitions name 'Titanic - Machine Learning from Disaster'
                    \nThe purpose of this web app is show how hyperparameter effect the model.
                """)
    show_metrics = st.checkbox("Show model information", False, key='show_metrics')

    

    st.sidebar.title("Model setting")
    st.sidebar.subheader('Choose Classifier')
    classifier = st.sidebar.selectbox('Classifier', ('Decision Tree', 'Random Forest'))
    
        
    if classifier == 'Decision Tree':
        st.sidebar.subheader("Model Hyperparameters")
        min_samples_split = st.sidebar.number_input("Minimum samples split", 2, 20, step=1, key='min_samples_split')
        min_samples_leaf = st.sidebar.number_input("Minimum samples leaf", 1, 20, step=1, key='min_samples_leaf')
        criterion =  st.sidebar.radio("criterion", ("gini", "entropy", "log_loss"), key="criterion")

    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("n_estimators", 10, 200, step=1, key='n_estimators')
        min_samples_split = st.sidebar.number_input("min_samples_split", 2, 20, step=1, key='min_samples_split')
        min_samples_leaf = st.sidebar.number_input("Minimum samples leaf", 1, 20, step=1, key='min_samples_leaf')
        criterion =  st.sidebar.radio("criterion", ("gini", "entropy", "log_loss"), key="criterion")
     
    if classifier == 'Decision Tree':
        model = DecisionTreeClassifier(min_samples_split=min_samples_split, 
                                           min_samples_leaf=min_samples_leaf,
                                           criterion=criterion)
    else:
        model = RandomForestClassifier(n_estimators=n_estimators,
                                           min_samples_split=min_samples_split, 
                                           min_samples_leaf=min_samples_leaf,
                                           criterion=criterion)
        
    pl = Pipeline(steps=[
        ('preprocessor', preprocessor), 
        ('model', model)
    ])

    pl.fit(X_train, y_train)
    accuracy = pl.score(X_test, y_test)
    y_pred = pl.predict(X_test)
    cm_dp, rou_dp = plot_metrics(y_pred, y_test)

    st.subheader(classifier +" Results")
    st.write("Model accuracy : " + str(accuracy))

    if show_metrics:
        if classifier == 'Decision Tree': 
            tree.plot_tree(model)
            st.pyplot()
        
        st.subheader('Confusion Matrix')
        cm_dp.plot()
        st.pyplot()

        st.subheader('ROC Curve')
        rou_dp.plot()
        st.pyplot()



if __name__ == '__main__':
    main()