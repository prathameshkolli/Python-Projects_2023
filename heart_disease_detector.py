# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset (assuming you have a CSV file named 'heart.csv')
@st.cache  # Caching for improved performance
def load_data():
    return pd.read_csv('heart.csv')

# Sidebar with user input for model training
st.sidebar.header('User Input')
def user_input_features():
    age = st.sidebar.slider('Age', 29, 77, 55)
    sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
    cp = st.sidebar.slider('Chest Pain Type (cp)', 0, 3, 1)
    trestbps = st.sidebar.slider('Resting Blood Pressure (mm Hg)', 94, 200, 120)
    chol = st.sidebar.slider('Serum Cholesterol (mg/dl)', 126, 564, 240)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'])
    restecg = st.sidebar.slider('Resting Electrocardiographic Results (restecg)', 0, 2, 1)
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved (thalach)', 71, 202, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina (exang)', ['No', 'Yes'])
    oldpeak = st.sidebar.slider('ST Depression Induced by Exercise Relative to Rest (oldpeak)', 0.0, 6.2, 1.0)
    slope = st.sidebar.slider('Slope of the Peak Exercise ST Segment (slope)', 0, 2, 1)
    ca = st.sidebar.slider('Number of Major Vessels Colored by Fluoroscopy (ca)', 0, 4, 0)
    thal = st.sidebar.slider('Thalassemia (thal)', 0, 3, 1)
    
    # Convert categorical variables to numerical
    sex = 1 if sex == 'Male' else 0
    fbs = 1 if fbs == 'Yes' else 0
    exang = 1 if exang == 'Yes' else 0
    
    user_data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    return user_data

user_data = user_input_features()

# Load the dataset
df = load_data()

# Split the dataset into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions
prediction = model.predict([list(user_data.values())])

# Display the prediction
st.title('Heart Disease Detector')
st.write('### User Input Features:')
st.write(user_data)
st.write('---')
st.write('### Prediction:')
st.write('**Probability of Heart Disease:**', prediction[0])

# Model evaluation on test set
y_pred = model.predict(X_test)
st.write('---')
st.write('### Model Evaluation on Test Set:')
st.write('**Accuracy:**', accuracy_score(y_test, y_pred))
st.write('**Classification Report:**')
st.write(classification_report(y_test, y_pred))
