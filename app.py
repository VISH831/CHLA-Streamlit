import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import pickle
from titlecase import titlecase
import time 

def load_data(file_path):
    return pd.read_csv(file_path)

# Load data and parse dates
df = load_data("CHLA_clean_data_2024_Appointments.csv")
df['APPT_DATE'] = pd.to_datetime(df['APPT_DATE'], format='%m/%d/%y %H:%M')

def main():
    st.markdown("""
    <style>
        .centered-text {
            text-align: center;
            font-size: 24px; /* or any other size */
            font-weight: bold; /* if you want it bold */
        }
    </style>
    <div class="centered-text">CHLA Resource</div>
    """, unsafe_allow_html=True)
    
    html_temp = """
    <div style="background:#22333b ;padding:10px">
    <h2 style="color:white;text-align:center;">Appointment No Show Prediction App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.info("Please fill in the details of the appointment to predict if the patient will attend")

    ### Date inputs
    col1, col2 = st.columns([1,1])
    min_date = df['APPT_DATE'].min().date()
    max_date = df['APPT_DATE'].max().date()
    with col1:
        start_datetime = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
    with col2:
        end_datetime = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

    start_datetime = pd.to_datetime(start_datetime)
    end_datetime = pd.to_datetime(end_datetime)

    if start_datetime > end_datetime:
        st.error("End Date should be after Start Date")

    ### Filter df by date inputs
    if start_datetime and end_datetime:
        mask = (df['APPT_DATE'] >= start_datetime) & (df['APPT_DATE'] <= end_datetime)
        fdf = df[mask]

    if not fdf.empty:
        
        ### Select and filter fdf by clinic
        clinic_selector = st.multiselect("Select a Clinic", fdf['CLINIC'].unique())
        if clinic_selector:
            fdf = fdf[fdf['CLINIC'].isin(clinic_selector)]

        if not fdf.empty:
            ### Prepare data for prediction
            pdf = fdf.copy() 
            pdf = pdf[[
                'AGE', 'CLINIC', 'TOTAL_NUMBER_OF_CANCELLATIONS', 'LEAD_TIME', 'TOTAL_NUMBER_OF_RESCHEDULED', 'TOTAL_NUMBER_OF_NOSHOW',
                'TOTAL_NUMBER_OF_SUCCESS_APPOINTMENT', 'HOUR_OF_DAY', 'NUM_OF_MONTH'
            ]]

            ### Load and run the predictor model
            model = pickle.load(open('model.pkl', 'rb'))

            ### Label encoding for categorical data
            le = LabelEncoder()
            if 'CLINIC' in pdf.columns:
                le.fit(df['CLINIC'].unique())
                pdf['CLINIC'] = le.transform(pdf['CLINIC'])

            ### Predict if there is data to predict on
            if st.button("Predict") and not pdf.empty:
                with st.spinner('Processing... Please wait'):
                    time.sleep(2)
                    predictions = model.predict(pdf)
                    probabilities = model.predict_proba(pdf)[:, 1]  # Assuming the second column is the probability of 'YES'
                    fdf['NO SHOW (Y/N)'] = ['YES' if x == 1 else 'NO' for x in predictions]
                    fdf['Probability'] = probabilities
                    st.success('Complete')

                # Display results
                # Convert MRN to string
                fdf['MRN'] = fdf['MRN'].astype(str)
                fdf['APPT_ID'] = fdf['APPT_ID'].astype(str)

                # Extract Date and Time from APPT_DATE
                fdf['Date'] = fdf['APPT_DATE'].dt.date
                fdf['Time'] = fdf['APPT_DATE'].dt.time

                st.dataframe(fdf[['MRN', 'APPT_ID', 'Date', 'Time', 'CLINIC', 'NO SHOW (Y/N)', 'Probability']])

                # Download link for the predictions
                csv = fdf.to_csv(index=False)
                st.download_button(
                    label="Download predictions as CSV",
                    data=csv,
                    file_name='predictions.csv',
                    mime='text/csv',
                )
            else:
                st.error("Prediction button not pressed")
        else:
            st.error("No appointments found for the selected clinics in the given date range")
    else:
        st.warning("Please select both a start and end date")

if __name__ == '__main__':
    main()
