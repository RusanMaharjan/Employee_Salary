import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

features = [
    'EngagementSurvey', 'EmpSatisfaction', 'Absences'
]

def model_train():
    df = pd.read_csv('HR_Dataset Refresh.csv')

    X = df[features]
    Y = df['Salary']

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, Y_train)

    return model, features, X, Y