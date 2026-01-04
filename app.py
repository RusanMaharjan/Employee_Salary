import streamlit as st
import pandas as pd
from model import model_train
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Salary Prediction")
st.subheader("Salary Prediction using Linear Regression")

model, features, X, Y = model_train()

st.sidebar.header("Input Features")

engagement_survey =st.sidebar.slider(
    'Engagement Survey',
    max_value=5.0,
    min_value=1.0,
    value=3.0,
    step=0.1
)

emp_satisfaction =st.sidebar.slider(
    'Employee Satisfaction',
    max_value=5,
    min_value=1,
    value=3,
    step=1
)

absences =st.sidebar.slider(
    'Absences',
    max_value=30,
    min_value=0,
    value=5,
    step=1
)

if st.sidebar.button("Predict Salary"):
    input_survey = pd.DataFrame([[
        engagement_survey, emp_satisfaction, absences
    ]],columns=features)

    prediction = model.predict(input_survey)[0]

    st.success("Prediction Successful!!")
    st.metric(
        label = "Predicted Salary",
        value = f"Rs. {prediction:,.2f}"
    )

# Visualization
st.header("Visualizations")
st.subheader("Engagement Survey vs. Salary")

fig, ax = plt.subplots()
sns.regplot(
    x = X['EngagementSurvey'],
    y=Y,
    scatter_kws={'color': 'blue', 'alpha':0.7},
    line_kws={'color':'red'},
    ax = ax
)
ax.set_xlabel("Engagement Survey")
ax.set_ylabel("Salary")
st.pyplot(fig)

fig, ax = plt.subplots()
sns.regplot(
    x = X['EmpSatisfaction'],
    y=Y,
    scatter_kws={'color': 'blue', 'alpha':0.7},
    line_kws={'color':'red'},
    ax = ax
)
ax.set_xlabel("Employee Satisfaction")
ax.set_ylabel("Salary")
st.pyplot(fig)


fig, axes = plt.subplots(2, 2, figsize=(14, 16))
axes = axes.flatten()

for i, feature in enumerate(features):
    sns.regplot(
        x = X[feature],
        y = Y,
        scatter_kws={'color': 'blue', 'alpha':0.7},
        line_kws={'color':'red'},
        ax = axes[i]
    )
    axes[i].set_title(f"{feature} vs. Salary")
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("Salary")

for j in range(len(features), len(axes)):
    fig.delaxes(axes[j])

st.pyplot(fig)

