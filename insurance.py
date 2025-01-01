import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib

# Custom Styling for Background, Fonts, and Layout
st.markdown("""
    <style>
    /* Global styling for background */
    .main {
        background-color: #f4f8fc;  /* Light blue background */
        font-family: 'Arial', sans-serif;
        padding: 20px;
    }

    /* Custom sidebar styling */
    .sidebar .sidebar-content {
        background-color: #1f77b4;  /* Sidebar color */
        color: white;
        padding: 20px;
    }

    /* Header styling */
    h1, h2, h3 {
        color: #2d3e50;
        font-family: 'Arial', sans-serif;
        font-weight: 700;
        text-align: center;
    }

    h1 {
        font-size: 3rem;
        color: #2c3e50;  /* Darker color for title */
        margin-bottom: 20px;
    }

    h2 {
        font-size: 2rem;
        margin-bottom: 10px;
    }

    h3 {
        font-size: 1.5rem;
    }

    /* Button Styling */
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
    }

    /* Add shadow to the plot areas */
    .stPlot {
        box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.1);
        border-radius: 10px;
        padding: 15px;
    }

    /* Add margin between elements */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Footer Styling */
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        padding: 10px;
        background-color: #1f77b4;
        color: white;
        text-align: center;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# Load dataset
df = pd.read_csv('insurance.csv')

# App title and description
st.title("Insurance Expense Prediction")
st.write("This app predicts medical insurance expenses based on user input.")

# Sidebar for user input
st.sidebar.header("📝 User Input")
age = st.sidebar.slider("Age", 18, 100, 25)
sex = st.sidebar.selectbox("Sex", ['male', 'female'])
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.sidebar.slider("Number of Children", 0, 5, 0)
smoker = st.sidebar.selectbox("Smoker", ['yes', 'no'])
region = st.sidebar.selectbox("Region", ['northeast', 'northwest', 'southeast', 'southwest'])

# Model training
X = df.drop(columns=['expenses'])
y = df['expenses']
numerical_features = ['age', 'bmi', 'children']
categorical_features = ['sex', 'smoker', 'region']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42, n_estimators=100))
])
model.fit(X_train, y_train)

# Predict insurance expense
st.header("💡 Predict Insurance Expense")

# Button for prediction
if st.button("🔮 Predict"):
    input_data = pd.DataFrame([[age, sex, bmi, children, smoker, region]],
                              columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
    
    # Predict the expense
    prediction = model.predict(input_data)
    
    st.success(f"💸 Predicted Insurance Expense: ${prediction[0]:,.2f}")

# Footer for app credits or additional info
st.markdown(
    """
    <div class="footer">
        <p>Built with ❤️ by Your Name | Powered by Streamlit</p>
    </div>
    """, unsafe_allow_html=True)
