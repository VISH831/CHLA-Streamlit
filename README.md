# CHLA No-show Prediction Model Deployment

## Overview
This repository contains the code and deployment setup for Project #2 - a machine learning model aimed at predicting patient no-shows at CHLA, deployed using Streamlit. The objective is to accurately predict whether a patient will show up for their scheduled appointment using historical CHLA data.

## Dataset
The CHLA dataset from Project 01 (P01) was utilized.

## Model Development
A variety of machine learning algorithms and hyperparameters were explored to select the best-performing model. The Random Forest Classifier was chosen due to its robustness and performance across several metrics crucial to the no-show prediction problem. Emphasis was placed on recall, precision, F1-score, and ROC-AUC, given the high costs associated with false predictions. <br>
<br>
Models tested: <br>
- Random Forest Classifier
- Gradient Boost Classifier
- AdaBoost Classifier

## Feature Selection
Top predictive features were carefully selected based on their impact on model performance, with a methodical approach involving correlation analysis, feature importance ranking, and domain knowledge. 

## Deployment
The final model was deployed using Streamlit in two different modes:
- Local Machine
- Github + Streamlit Server

## Public URL
The model is deployed and publicly accessible at the following Streamlit Server URL: <br>
https://appointmentpredictiontool.streamlit.app/

## Repository Contents
- app.py - contains the code for the Streamlit deployed app.
- project2Final.ipynb - contains the code to explore the data, test the different models, and save the best final model.
- model.pkl - pickle file of the saved best model.
- encoder.pkl - pickle file of the encoded features from the notebook file.
- requirements.txt - the package requirements to deploy and run the model and app.

## Getting Started
To run the Streamlit app locally, clone the repository, install the dependencies, and execute the Streamlit run command:
```sh
git clone https://github.com/xvmarquez/AppointmentPredictionTool.git
cd AppointmentPredictionTool
pip install -r requirements.txt
streamlit run app.py
