import streamlit as st
import pandas as pd
import numpy as np
import pickle
import category_encoders as ce
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import LabelEncoder


df = pd.read_parquet('bank-full.parquet', engine="fastparquet")
df_edu = pd.read_csv('edu_without_unknown.csv')
df_job = pd.read_csv('job_without_unknown.csv')
# df_data = pd.read_excel('bank-full.xlsx')

unique_jobtype = df["job"].unique().tolist()
unique_jobtype = [job for job in unique_jobtype if job != 'unknown']

unique_days = df["day"].unique().tolist()

unique_month = df["month"].unique().tolist()

unique_duration = df["duration"].unique().tolist()


# with open("age_scaler.pkl", "rb") as f:
#     age_scaler = pickle.load(f)
age_scaler = joblib.load('age_scaler.joblib')
balance_scaler = joblib.load('balance_scaler.joblib')
day_scaler = joblib.load('day_scaler.joblib')
duration_scaler = joblib.load('duration_scaler.joblib')
education_lb = joblib.load('education_lb.joblib')
month_lb = joblib.load('month_lb.joblib')
label_encoder = joblib.load("y_encoder.joblib")

model = joblib.load("xgb_model.joblib")
# Streamlit UI
st.set_page_config(page_title="A Portuguese Banking Institution", layout="wide")

st.markdown("""
    <style>
        /* Make sidebar red */
        [data-testid="stSidebar"] {
            background-color: #ff4b4b;  /* Red background */
        }

        /* Style for sidebar titles */
        .sidebar-title {
            color: white;
            font-weight: bold;
            font-size: 18px;
            padding: 5px 10px;
            background-color: #ff4b4b;
            border-radius: 5px;
            text-align: center;
            width: 100%;
            display: block;
        }

        /* Make buttons full width */
        div.stButton > button {
            width: 100%;
            color: #fff;
            background-color: #e60000;
            border: none;
            border-radius: 8px;
            padding: 0.5em;
            margin: 0.3em 0;
        }

        div.stButton > button:hover {
            background-color: #cc0000;
        }
    </style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown('<div class="sidebar-title">🔍 Navigation</div>', unsafe_allow_html=True)

    # Initialize page state if not already set
    if "page" not in st.session_state:
        st.session_state.page = "Problem intro"

    # Navigation buttons
    if st.sidebar.button("Problem intro"):
        st.session_state.page = "Problem intro"

    if st.sidebar.button("Insights and Charts"):
        st.session_state.page = "Insights and Charts"

    if st.sidebar.button("Predict Subscriptions"):
        st.session_state.page = "Predict Subscriptions"

    if st.sidebar.button("Model Summary page"):
        st.session_state.page = "Model Summary page"


# page = st.sidebar.radio("Navigation", ["Problem intro", "Insights and Charts",'Predict Subscriptions','Model Summary page'])

if st.session_state.page == "Problem intro":
 
        st.title("📈 Bank Term Deposit Subscription Prediction")
        st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT72gTMjYCp2d5xuoyRMJ37kJrgn0m6JXUK4w&s',
          use_container_width = True)
        
        st.markdown("---")

        # Problem Statement
        st.markdown("""
        ### 🌟 Problem Statement

        Imagine you are the marketing head of a prestigious Portuguese bank.  
        Each day, your team tirelessly calls hundreds of clients, hoping they’ll subscribe to a **term deposit**.  
        But the success rate? Frustratingly low.

        **What if you could predict** who is likely to say "**yes**" even before you dial?

        🔍 **Objective:**  
        The bank has conducted multiple marketing campaigns through phone calls. Now, they want to **leverage machine learning** to **predict** whether a client will **subscribe to a term deposit** based on their profile and past campaign interactions.

        This project focuses on:
        - 📚 **Analyzing** historical marketing data
        - 🔮 **Building a predictive model** to classify clients as "yes" or "no"
        - 🌐 **Deploying** this model as an **interactive web application** using **Streamlit**
        - 💰 **Empowering marketing teams** to optimize targeting, enhance campaign performance, and reduce costs

        ---
        """)

        # Why this matters
        st.markdown("""
        ### 🚀 Why This Matters

        - **Smarter Targeting:** Focus efforts on clients more likely to subscribe.
        - **Higher Conversion Rates:** Boost campaign success dramatically.
        - **Cost Efficiency:** Save time, money, and resources.

        > *"Marketing is no longer about the stuff you make but about the stories you tell."*  
        > — **Seth Godin**

        With the power of **data** and **machine learning**, let's help the bank tell the right story to the right client! 📢

        ---
        """)

        # Footer message
        st.markdown("""
        **⬇️ Use the navigation sidebar to explore the model, make predictions, and dive into the data!**
        """)


elif st.session_state.page == "Insights and Charts":
   
    st.title('Exploratory Data Analysis')

    # Create Tabs
    tab1, tab2, tab3 = st.tabs(["🧍 Client Demographics", "🏦 Financial Status", "📞 Marketing Interaction"])

    # Tab 1 - Client Demographics
    with tab1:
        st.header("Client Demographics Insights")

        st.subheader("Age Group vs Subscription")
        age_bins = pd.cut(df['age'], bins=[0, 20, 30, 40, 50, 60, 70, 90])

        # Plot
        fig, ax = plt.subplots(figsize=(8,6))  # <-- Important to create a figure for Streamlit
        sns.countplot(x=age_bins, hue='y', data=df, ax=ax)
        ax.set_title("Target Distribution Across Age Groups")
        ax.set_xlabel("Age Group")
        ax.set_ylabel("Number of Clients")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Show plot in Streamlit
        st.pyplot(fig)

        st.write("### Insights")

        st.markdown("""
        <ul>
        <li> Clients aged 30-40 years show the highest subscription rate.</li>
        <li> Clients aged 20-30 years show slight interest in subscription.</li>
        <li> Focus marketing efforts on clients aged 20-40 for the best results.</li>
        </ul>
        """, unsafe_allow_html=True)


        st.subheader("Education Type vs Subscription")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.countplot(x='education', hue='y', data=df_edu)

    # Set the title and labels
        plt.title("Which Education Type is Likely to Subscribe?")
        
        plt.xlabel("Education")
        plt.ylabel("Count")
        st.pyplot(fig)
        
        st.write("### Insights")

        st.markdown("""
        <ul>
        <li> Clients who subscribed a lot belongs to secondary education type comparing to other education type clients.</li>
        <li> Focus marketing efforts on clients in above education type for the best results.</li>
        </ul>
        """, unsafe_allow_html=True)


        st.subheader("Job Type vs Subscription")

        fig, ax = plt.subplots(figsize=(8,6))
        plt.xticks(rotation = 30)
        sns.countplot(x='job', hue='y', data=df_job)
        plt.title("Job type vs Target (Yes/No)")
        plt.xlabel("what kind of job people have subscribed term deposit?")
        plt.ylabel("Count")
        st.pyplot(fig)

        st.write("### Insights")

        st.markdown("""
        <ul>
        <li> Most subscribed top 3 clients jobs are Blue-collar - First priority, 
                    Technician - Second priority, Management - Third priority.</li>
        <li> These Clients subscriptions for term deposit is thriving.</li>
        <li> Focus marketing efforts on these clients for the best results.</li>
        </ul>
        """, unsafe_allow_html=True)

        st.subheader("Marital Status vs Subscription")
        fig, ax = plt.subplots(figsize=(8,6))

        sns.countplot(x='marital', hue='y', data=df)
        plt.title("what marital status  customers likely to subscribe?")
        plt.xlabel("Marital status")
        plt.ylabel("Counts")
        st.pyplot(fig)
        
        st.write("### Insights")

        st.markdown("""
        <ul>
        <li> Clients who were Married and Single show more interest in Term Subscription.</li>
        <li> These Clients subscriptions for term deposit is thriving.</li>
        <li> Focus marketing efforts on these clients for the best results.</li>
        </ul>
        """, unsafe_allow_html=True)



    # Tab 2 - Financial Status
    with tab2:
        st.header("Client Financial Status")

        st.subheader("Personal Loan / Home loan vs Subscription")
        fig, ax = plt.subplots(figsize=(8,6))

        cross_tab = pd.crosstab([df['loan'], df['housing']], df['y'], normalize='index')
        sns.heatmap(cross_tab, annot=True, cmap="YlGnBu", fmt=".2f")
        plt.title('Relationship between Personal Loan, Housing and Subscription (y)')
        plt.ylabel('Personal Loan + Housing Combination')
        plt.xlabel('Subscribed (y)')
        st.pyplot(fig)
            
        st.write("### Insights")

        st.markdown("""
        <ul>
        <li> Clients who weren't taken any home loan and personal loan were prefer to subscribe.</li>
        <li> Focus marketing efforts on these clients for the best results.</li>
        </ul>
        """, unsafe_allow_html=True)

        # Chart for home loan
        st.subheader("Credit Default vs Subscription")

        fig, ax = plt.subplots(figsize=(8,6))
        sns.countplot(x='default', hue='y', data=df)
        plt.title('Has credit in default? (binary: "yes","no") vs Target (Yes/No)')
        plt.xlabel("loan paid/unpaid customers")
        plt.ylabel("Count")
        st.pyplot(fig)

        st.write("### Insights")

        st.markdown("""
        <ul>
        <li> Give lot of attention to no credit in default status.</li>
        <li> Focus marketing efforts on these clients for the best results.</li>
        </ul>
        """, unsafe_allow_html=True)

        # Chart for credit in default

    # Tab 3 - Marketing Interaction
    with tab3:
        st.header("Marketing Campaign Analysis")
        st.subheader("Subscription Trend by Month/Season")
        fig, ax = plt.subplots(figsize=(8,6))

        sns.countplot(x='month', hue='y', data=df)
        plt.title("Month vs Target (Yes/No)")
        plt.xlabel("which month have lot of subscriptions?")
        plt.ylabel("Count")
        st.pyplot(fig)

        st.write("### Insights")

        st.markdown("""
        <ul>
        <li> Seasonal trends are May , August , July mostly in these lot of subscriptions happened.</li>
        <li> Adapt these trends and make campaigns as per this.</li>
        </ul>
        """, unsafe_allow_html=True)

    

    

elif st.session_state.page == "Predict Subscriptions":


    @st.cache_data
    def transform_inputs(age, balance, day, month, duration, job, marital_status, education, default, housing, personal_loan):
        # Apply transformations
        age_scaled = age_scaler.transform([[age]])[0][0]
        balance_scaled = balance_scaler.transform([[balance]])[0][0]
        day_scaled = day_scaler.transform([[day]])[0][0]
        month_encode = [month_lb.transform([month])[0]]
        duration_scaled = duration_scaler.transform([[duration]])[0][0]
        education_encode = [education_lb.transform([education])[0]]
        
        # One-hot encoding for categorical features
        job_input = {f'job_{j}': 1 if j == job else 0 for j in unique_jobtype}
        marital_input = {f'marital_{m}': 1 if m == marital_status else 0 for m in ['divorced', 'married', 'single']}
        default_input = {'default_no': 1 if default == 'no' else 0, 'default_yes': 1 if default == 'yes' else 0}
        housing_input = {'housing_no': 1 if housing == 'no' else 0, 'housing_yes': 1 if housing == 'yes' else 0}
        personal_loan_input = {'loan_no': 1 if personal_loan == 'no' else 0, 'loan_yes': 1 if personal_loan == 'yes' else 0}

        # Combine all into a single dictionary
        data = {
            'age': age_scaled,
            'education': education_encode[0],
            'balance': balance_scaled,
            'day': day_scaled,
            'month': month_encode[0],
            'duration': duration_scaled,
            **job_input,
            **marital_input,
            **default_input,
            **housing_input,
            **personal_loan_input
        }

        return data


    st.title('Predicting Term Deposit Subscription')

    age = st.number_input('Enter the age', value=0)
    job = st.selectbox('Select Job', unique_jobtype)
    marital_status = st.selectbox('Enter the marital status', ['divorced', 'married', 'single'])
    education = st.selectbox('Enter the education', ['primary', 'secondary', 'tertiary'])
    default = st.selectbox('Are you in credit in default status?', ['yes', 'no'])
    balance = st.number_input('Average yearly balance worth?', value=0.0)
    housing = st.selectbox('Already homeloan taken person?', ['yes', 'no'])
    personal_loan = st.selectbox('Already personal loan taken person?', ['yes', 'no'])
    day = st.selectbox('Last contact day of the month?', unique_days)
    month = st.selectbox('Last contact month of the year?', unique_month)
    duration = st.selectbox('Last contact duration in secs', [f"{dur} secs" for dur in unique_duration])
    selected_duration_value = int(duration.split()[0])

    input_data = transform_inputs(age, balance, day, month, selected_duration_value, job, marital_status, education, default, housing, personal_loan)

    # Convert input data into a DataFrame
    input_dataframe = pd.DataFrame([input_data])

    model_features = model.get_booster().feature_names  # Get the feature names from the model
    input_dataframe = input_dataframe[model_features] 
    # st.write("Model expects features:", model_features)
    # st.write("Input features available:", input_dataframe)


    input_dataframe = input_dataframe.apply(pd.to_numeric, errors='coerce')
    # st.write("Any NaNs in input?", input_dataframe.isnull().sum())
    # st.write("Month classes:", month_lb.classes_)
    # st.write("Education classes:", education_lb.classes_)

    if st.button("Predict"):
    # Get predicted probabilities (probabilities of class 1)
            proba = model.predict_proba(input_dataframe)[0][1]  # [1] gives the probability of class 1
    
    # Set a custom threshold (for example, 0.7)
            threshold = 0.5
            if proba >= threshold:
                st.success("As per the above details this client will make subscription")
            else:
                st.success("As per the above details this client will not make subscription")


      

else:
    if st.session_state.page == "Model Summary page":

        st.title("📈 Model Summary - XGBoost Classifier")

        # 1. Model Description
        st.header("1. Model Description")
        st.write("""
        We have used the **XGBoost Classifier** for a binary classification task to predict the target variable `y`.
        XGBoost is chosen due to its powerful gradient boosting framework that handles large datasets and provides high predictive performance.
        """)

        # 2. Input Features
        st.header("2. Input Features")
        st.write("""
        The following features were used for training the model:
        - **age**
        - **job**
        - **marital**
        - **education**
        - **default**
        - **balance**
        - **housing**
        - **loan**
        - **day**
        - **month**
        - **duration**
        """)

        # 3. Model Performance Metrics
        st.header("3. Model Performance Metrics")
        st.write('Acurracy = 86.17%')
        st.write("Precision is 44.66% note:Because the test data set have more 0th class(no) when comparing to 1st class")
        st.write("Recall is 75.92%")
        st.write("F1 score is 56.24%")
        st.write("ROC-AUC Score is 82%")

        # 4. Confusion Matrix
        st.header("4. Confusion Matrix")
        confusion_img = Image.open("confusion_matrix.png")
        st.image(confusion_img, caption="Confusion Matrix", use_container_width=True)

        # 5. ROC Curve
        st.header("5. ROC Curve")
        roc_img = Image.open("roc_curve.png")
        st.image(roc_img, caption="ROC-AUC Curve", use_container_width=True)


        # 7. Limitations and Future Work
        st.header("6. Limitations and Future Work")
        st.write("""
        - Model can be further improved by feature engineering (e.g., creating interaction terms).
        - Hyperparameter tuning could push the ROC-AUC score even higher.
        - Consider testing on cross-validation folds for more robustness.
        """)

            
        

    