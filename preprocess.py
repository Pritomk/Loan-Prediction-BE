import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def fill_nan_data(df):
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mean())
    df['Gender'] = df["Gender"].fillna(df['Gender'].mode()[0])
    df['Married'] = df["Married"].fillna(df['Married'].mode()[0])
    df['Dependents'] = df["Dependents"].fillna(df['Dependents'].mode()[0])
    df['Self_Employed'] = df["Self_Employed"].fillna(df['Self_Employed'].mode()[0])
    return df

def data_pre_process(df):
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['ApplicantIncomeLog'] = np.log(df['ApplicantIncome']+1)
    df['CoapplicantIncomeLog'] = np.log(df['CoapplicantIncome']+1)
    df['LoanAmountLog'] = np.log(df['LoanAmount']+1)
    df['Loan_Amount_Term_Log'] = np.log(df['Loan_Amount_Term']+1)
    df['Total_Income_Log'] = np.log(df['Total_Income']+1)
    return df

def drop_unnecessary_data(df):
    cols = ['ApplicantIncome', 'CoapplicantIncome', "LoanAmount", "Loan_Amount_Term", "Total_Income", 'Loan_ID', 'CoapplicantIncomeLog']
    df = df.drop(columns=cols, axis=1)
    return df

def label_encoding(df):
    cols = ['Gender',"Married","Education",'Self_Employed',"Property_Area","Dependents"]
    le = LabelEncoder()
    for col in cols:
        df[col] = le.fit_transform(df[col])
    return df

def all_preprocess(df):
    df = fill_nan_data(df)
    df = data_pre_process(df)
    df = drop_unnecessary_data(df)
    df = label_encoding(df)
    return df