# Visualizes Tree-Based Models via Streamlit Project

## This is a Streamlit project that visualizes 2 Tree-Based Models, namely Decision tree and Random forest model. This project aims to illustrate how hyperparameter effect model decision.
  
- ![image](https://github.com/KiattiphumSuw/Titanic-Streamlit-project/assets/83391695/9d9ac211-48b5-4aa4-883c-827cb171d39a)

This project is an example that build model with data pipeline in python and show performance via web app by using Streamlit. This project composite of these element.

### ETL process
* create data pipeline with sklearn pipeline.
* filling numerical null value with SimpleImputer (using constant strategy).
* filling categorical null value with SimpleImputer (using constant strategy).
* encode categorical value with OneHotEncoder.

### Metrics
* both Decision tree and Random forest model show Confusion Matrix ROC Curve metrics.
* Decision tree also visualize its tree.

## How to install this project
1. clone this project
2. Run ```pip install streamlit``` in the case that you haven't installed Streamlit
3. Run ```streamlit run app.py``` to start programe in web app
