import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from openai import OpenAI
import utils as ut

client = OpenAI(
  base_url = "https://api.groq.com//openai/v1",
  api_key = os.environ.get("GROQ_API_KEY")
)


#Function to load our models
def load_model(filename):
  with open(filename, 'rb') as file:
    return pickle.load(file)

#Load all models
xgboost_model = load_model('xgb_model.pkl')

naive_bayes_model = load_model('nb_model.pkl')

svm_model = load_model('svm_model_prob.pkl')

rf_model = load_model('rf_model.pkl')

dt_model = load_model('dt_model.pkl')

knn_model = load_model('knn_model.pkl')

voting_clf_model = load_model('voting_clf_hard.pkl')

xgboost_smote_model = load_model('xgb_smote.pkl')

xgboost_feature_eng_model = load_model('xgb_featureEngineered.pkl')

#Helper fucntion to prepare input data for the models
def prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary):
  #All into a dict
  input_dict = {
    'CreditScore' : credit_score,
    'Age': age,
    'Tenure': tenure, 
    'Balance': balance,
    'NumOfProducts': num_products, 
    'HasCrCard': int(has_credit_card),
    'IsActiveMember': int(is_active_member),
    'EstimatedSalary':estimated_salary,
    'Geography_France': 1 if location == "France" else 0,
    'Geography_Germany': 1 if location == "Germany" else 0,
    'Geography_Spain': 1 if location == "Spain" else 0,
    'Gender_Male': 1 if gender == "Male" else 0,
    'Gender_Female': 1 if gender =="Female" else 0
  }
  input_df = pd.DataFrame([input_dict])
  return input_df, input_dict

#Function that will make the prediction using the input DF and input dict
def make_prediction(input_df, input_dict):

  probabilities = {
    'XGBoost': xgboost_model.predict_proba(input_df)[0][1],
    'Random Forest': rf_model.predict_proba(input_df)[0][1],
    'K-Nearest Neighbors': knn_model.predict_proba(input_df)[0][1],
    "SVM" : svm_model.predict_proba(input_df)[0][1],
  }

  avg_probabilities = np.mean(list(probabilities.values()))

  col1, col2 = st.columns(2)

  with col1:
    fig = ut.create_gauge_chart(avg_probabilities)
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"The customer has a {avg_probabilities*100:.2f} probability of churning.")

  with col2:
    fig_probs = ut.create_model_prob_chart(probabilities)
    st.plotly_chart(fig_probs, use_container_width=True)


  return avg_probabilities

#Function that will generate teh explanation of the predictions for each customer using LLM-llama and a good detailed prompt
def explain_prediction(probability, input_dict, surname):

  prompt = f"""You are an expert scientist at a bank with more a PhD in Computer Science and with more than 10 years of experience in the financial sector and considered a pioner in data science, where you specialize in interpreting and explaining predictions of a machine learning models.

  Your machine learning model has predicted that a customer named {surname} has a {round(probability * 100, 1)}% probability of churning, based on the information provided below.

  Here is the customer information:
  {input_dict}

  Here is the machine learning model's top 10 most important features for predicting churn based on real data:

                  Feature | Imporatance
              --------------------------------
            NumOfProducts | 0.323888
          IsActiveMemeber | 0.164146
                      Age | 0.109550
       Geography_Gemrmany | 0.091373
                  Balance | 0.052786
         Geography_France | 0.046463
            Gender_Female | 0.045283
          Geogprahy_Spain | 0.036855
              CreditScore | 0.035005
          EstimatedSalary | 0.032655
                HasCrCard | 0.030054
                   Tenure | 0.030054
              Gender_male | 0.000000

          {pd.set_option('display.max_columns', None)}

          Here are the summary statistics for churned customers:
          {df[df['Exited'] == 1].describe()}


  - If the customer has over 40% risk of churning, generate a 3 sentence explanation of why they are at risk of churning.
  - If the customer has less than 40% risk of churning, generate a 3 sentence explanation of why they might not be at risk of churning.
  - Your explanation should be based on the customer's information, the summary statistics of churned and non-churned customers, and the feature importance provided.

    Don't mention the probability of churning, or the machine learning model, or say anything like "Based on the machine learning model's prediction and the top 10 most important features", just explain the prediction.
   Paragraph Structure: Responses should be structured in clear, distinct paragraphs without inline lists or excessive parentheses. Avoid dense, complex sentences that may be difficult to read.

   Consistent Style: Use consistent language and style throughout the response. Avoid overly technical jargon unless it's necessary, and keep sentences concise for clarity.

   No Line Breaks in Words or Numbers: Ensure words or numbers do not break across lines. Words like "101,348" or "91,108" should appear as a single unit without breaking or spacing issues.

   Logical Flow: Begin with a summary of the customer’s risk factors, then explain supporting factors in the next paragraphs. Each paragraph should address a specific point, such as risk indicators, mitigating factors, or final conclusions, in a logical sequence.

   Do not mention customer name or surname since we already know it.

  """
  print("EXPLANATION PROMPT", prompt)

  raw_response = client.chat.completions.create(
    model = "llama-3.2-3b-preview",
    messages = [{
      "role": "user",
      "content": prompt
    }],
  )
  return raw_response.choices[0].message.content

#Function that will send a personalize email to the customer
def generate_email(probability, input_dict, explanation, surname):
  prompt =f"""You are a manager at a LR bank. You are responsible for ensuring customers stay with the bank and are incentivized with various offers.

  You noticed a customer named {surname} has a {round(probability * 100, 1)}% probability of churning.

  Here is the customer information:
  {input_dict}

  Here is some explanation as to why the customer might be at risk of churning:
  {explanation}

  Generate an email to the customer based on their informatio, asking them to stay if they are at risk if churning, or offering them incentives so that they become more loyal to the bank.

  Make sure to list out a set of oncentives to stay based on their information, in bullet point format. Don't ever mention the probability of churning, or the machine learning model to the customer.
  """

  raw_response = client.chat.completions.create(
    model = "llama-3.1-8b-instant",
    messages =[{
      "role": "user",
      "content": prompt
    }],
  )

  print("\m\nEMAIL PROMPT", prompt)

  return raw_response.choices[0].message.content

#Start of the UI 
st.title("Customer Churn Prediction")


#1.Read CSV file 

df = pd.read_csv('churn.csv')

#2.Make a list to select customer and see datam using cutomer ID and their last name, a lambda function is used to select customerID and last name
customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]

#Display customer list
selected_customer_option = st.selectbox("Select Customer", customers)

#When a customer is selected, store id and surname into separate variables
#Filter all the data of the selected customer
if selected_customer_option:
  
  selected_customer_id = int(selected_customer_option.split(" - ")[0])

  selected_surname = selected_customer_option.split(" - ")[1]

  selected_customer = df.loc[df['CustomerId'] == selected_customer_id]

  col1, col2 = st.columns(2)
  #UI column 1 with data from csv
  with col1:
    credit_score = st.number_input(
      "Credit Score",
      min_value=300,
      max_value=850,
      value = int(selected_customer['CreditScore'])
    )

    location = st.selectbox(
      "Location", 
      ["Spain", "France", "Germany"],
      index = ["Spain", "France", "Germany"].index(selected_customer['Geography'].iloc[0])
    )
    
    gender = st.radio("Gender", ["Male", "Female"],
                     index=0 if selected_customer['Gender'].iloc[0] == 'Male' else 1)
    age = st.number_input(
      "Age", 
      min_value=18,
      max_value=100,
      value = int(selected_customer['Age'].iloc[0])
    )
    ternure = st.number_input(
      "Tenure (years)",
      min_value=0,
      max_value=50,
      value = int(selected_customer['Tenure'].iloc[0])
    )
  #UI column 2 with data from csv
  with col2:
    balance = st.number_input(
      "Balance",
      min_value=0.0,
      value = float(selected_customer['Balance'])
    )
    num_of_products = st.number_input(
      "Number of Products",
      min_value=1,
      max_value=10,
      value = int(selected_customer['NumOfProducts'])
    )
    has_credit_card = st.checkbox(
      "Has Credit Card",
      value = bool(selected_customer['HasCrCard'].iloc[0])
    )
    is_active_menber = st.checkbox(
      "Is Active Member",
      value = bool(selected_customer['IsActiveMember'].iloc[0])
    )
    estimated_salary = st.number_input(
      "Estimated Salary",
      min_value=0.0,
      value = float(selected_customer['EstimatedSalary'])
    )

  input_df, input_dict = prepare_input(credit_score, location, gender, age, ternure, balance, num_of_products, has_credit_card, is_active_menber, estimated_salary)

  ave_probability = make_prediction(input_df, input_dict)

  explanation = explain_prediction(ave_probability, input_dict, selected_customer['Surname'])


  st.markdown("---")

  st.subheader("Explanation of Prediction")

  st.markdown(explanation)

  email = generate_email(ave_probability, input_dict, explanation, selected_customer['Surname'])

  st.markdown("---")

  st.subheader("Personalized Email")

  st.markdown(email)
