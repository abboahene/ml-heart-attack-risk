import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.combine import SMOTETomek
import shap
import base64
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Heart Attack Risk Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E88E5;
    }
    .info-box {
        background-color: #F0F2F6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Main header
st.markdown("<h1 class='main-header'>❤️ Heart Attack Risk Prediction</h1>", unsafe_allow_html=True)

# App description
st.markdown("""
This app helps predict the risk of heart attack using machine learning models.
Upload your data or use our sample dataset to get started.
""")

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/heart-health.png", width=100)
st.sidebar.title("Controls")

# Functions for data preprocessing and modeling
@st.cache_data
def load_sample_data():
    try:
        df = pd.read_csv("heart-attack-risk-prediction-dataset.csv")
        return df
    except FileNotFoundError:
        st.error("Sample dataset not found. Please upload your own data.")
        return None

def preprocess_data(df):
    # Handle missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())
    
    # Convert categorical variables to dummies
    df_processed = pd.get_dummies(df, drop_first=True)
    
    return df_processed

def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(12, 8))
    corr = df.corr()
    mask = np.triu(corr)
    sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', ax=ax, vmin=-1, vmax=1)
    plt.title("Feature Correlation Heatmap")
    return fig

def train_models(X, y):
    # Class balance with SMOTETomek
    smote_tomek = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {}
    
    with st.spinner('Training XGBoost model...'):
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        xgb.fit(X_train_scaled, y_train)
        models['XGBoost'] = {
            'model': xgb,
            'predictions': xgb.predict(X_test_scaled),
            'probabilities': xgb.predict_proba(X_test_scaled)[:, 1]
        }
    
    with st.spinner('Training Random Forest model...'):
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train_scaled, y_train)
        models['Random Forest'] = {
            'model': rf,
            'predictions': rf.predict(X_test_scaled),
            'probabilities': rf.predict_proba(X_test_scaled)[:, 1]
        }
    
    return models, X_train, X_test, y_train, y_test, scaler, X_train_scaled, X_test_scaled

def plot_confusion_matrix(y_true, y_pred, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {title}")
    return fig

def plot_feature_importance(model, feature_names, title):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = np.array(feature_names)[indices]
    values = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=values[:15], y=features[:15], palette='viridis', ax=ax)
    plt.title(f'{title} Feature Importances')
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    return fig

def clinical_feature_analysis(model, feature_names):
    # Get feature importances
    importances = model.feature_importances_
    
    # Define modifiability categories
    modifiable = [
        'Cholesterol', 'Blood Pressure', 'BMI', 'Diet', 'Exercise', 
        'Smoking', 'Alcohol', 'Physical Activity', 'Stress Level',
        'Sedentary Hours', 'Sleep Hours'
    ]
    
    semi_modifiable = [
        'Medication Use', 'Obesity', 'Heart Rate', 'Triglycerides'
    ]
    
    non_modifiable = [
        'Age', 'Sex', 'Family History', 'Previous Heart Problems', 
        'Diabetes', 'Income'
    ]
    
    # Categorize each feature
    feature_categories = []
    for feature in feature_names:
        if any(factor in feature for factor in modifiable):
            category = 'Modifiable'
        elif any(factor in feature for factor in semi_modifiable):
            category = 'Semi-Modifiable'
        elif any(factor in feature for factor in non_modifiable):
            category = 'Non-Modifiable'
        else:
            category = 'Other'
        feature_categories.append(category)
    
    # Create DataFrame with feature info
    feature_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances,
        'Category': feature_categories
    })
    
    # Sort by importance
    feature_df = feature_df.sort_values('Importance', ascending=False)
    
    # Calculate importance by category
    category_importance = feature_df.groupby('Category')['Importance'].sum().reset_index()
    
    # Plot feature importance with clinical context
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', hue='Category', data=feature_df.head(15), ax=ax1)
    plt.title('Top 15 Feature Importances with Clinical Modifiability Context')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    # Plot categorical pie chart
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.pie(category_importance['Importance'], 
            labels=category_importance['Category'],
            autopct='%1.1f%%', 
            startangle=90,
            colors=['#2ecc71', '#f1c40f', '#e74c3c', '#95a5a6'])
    ax2.set_title('Contribution to Heart Attack Risk by Factor Modifiability')
    ax2.axis('equal')
    
    # Create actionability score
    actionability_score = (
        category_importance[category_importance['Category'] == 'Modifiable']['Importance'].sum() +
        0.5 * category_importance[category_importance['Category'] == 'Semi-Modifiable']['Importance'].sum()
    ) / category_importance['Importance'].sum()
    
    return feature_df, fig1, fig2, actionability_score

def predict_individual_risk(input_data, model, X_columns, scaler, explainer=None):
    df_input = pd.DataFrame([input_data])
    # Ensure columns match the training data
    df_input_dummies = pd.get_dummies(df_input)
    # Fill missing columns with 0
    for col in X_columns:
        if col not in df_input_dummies.columns:
            df_input_dummies[col] = 0
    # Select only columns that were in the training data
    df_input_dummies = df_input_dummies[X_columns]
    
    # Scale the data
    df_input_scaled = scaler.transform(df_input_dummies)
    
    # Make prediction
    prediction = model.predict(df_input_scaled)[0]
    proba = model.predict_proba(df_input_scaled)[0][1]
    
    # Create SHAP explanation if explainer exists
    if explainer is not None:
        try:
            shap_values = explainer(df_input_scaled)
            return prediction, proba, shap_values
        except:
            return prediction, proba, None
    
    return prediction, proba, None

# App navigation
app_mode = st.sidebar.selectbox(
    "Choose a mode",
    ["Home", "Data Exploration", "Model Training", "Risk Prediction"]
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'models' not in st.session_state:
    st.session_state.models = None
if 'X' not in st.session_state:
    st.session_state.X = None
if 'y' not in st.session_state:
    st.session_state.y = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'explainer' not in st.session_state:
    st.session_state.explainer = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None

# Home page
if app_mode == "Home":
    st.markdown("<h2 class='sub-header'>Welcome to the Heart Attack Risk Prediction App!</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### About this app
        This application helps healthcare professionals and individuals assess heart attack risk using machine learning.
        
        ### Key features:
        - Data exploration and visualization
        - Model training with XGBoost and Random Forest
        - Clinical context analysis of risk factors
        - Personalized risk prediction
        
        ### How to use:
        1. Start by uploading your dataset or use our sample data
        2. Explore the data visualizations
        3. Train and evaluate models
        4. Get personalized risk predictions
        """)
    
    with col2:
        st.image("https://img.icons8.com/color/240/000000/heart-with-pulse.png", width=200)
    
    # Data loading section
    st.markdown("<h3 class='sub-header'>Load Data</h3>", unsafe_allow_html=True)
    
    data_source = st.radio(
        "Choose data source:",
        ("Upload your own data", "Use sample data")
    )
    
    if data_source == "Upload your own data":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state.data = df
            st.success("Data successfully loaded!")
    else:
        sample_data = load_sample_data()
        if sample_data is not None:
            st.session_state.data = sample_data
            st.success("Sample data loaded!")
    
    if st.session_state.data is not None:
        st.markdown("<p class='info-box'>✅ Data loaded! Navigate to 'Data Exploration' to continue.</p>", unsafe_allow_html=True)

# Data Exploration page
elif app_mode == "Data Exploration":
    st.markdown("<h2 class='sub-header'>Data Exploration</h2>", unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("Please load data on the Home page first!")
    else:
        df = st.session_state.data
        
        # Data overview
        st.subheader("Dataset Overview")
        st.write(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.checkbox("Show raw data", value=True):
                st.write(df.head())
        with col2:
            if st.checkbox("Show column information", value=True):
                st.write(df.dtypes)
        
        # Missing values
        st.subheader("Missing Values Analysis")
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            st.write(missing_data[missing_data > 0])
        else:
            st.write("No missing values found in the dataset.")
        
        # Basic statistics
        st.subheader("Statistical Summary")
        st.write(df.describe())
        
        # Data distributions
        st.subheader("Data Distributions")
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Select column for histogram:", numeric_cols)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df[selected_col], kde=True, ax=ax)
            plt.title(f"Distribution of {selected_col}")
            st.pyplot(fig)
        
        # Target variable distribution
        st.subheader("Target Variable Distribution")
        
        target_options = ["Heart Attack Risk (Binary)", "Heart Attack Risk (Text)", "Heart Attack Risk"]
        target_col = next((col for col in target_options if col in df.columns), None)
        
        if target_col:
            fig, ax = plt.subplots(figsize=(8, 6))
            value_counts = df[target_col].value_counts()
            ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            plt.title(f"Distribution of {target_col}")
            st.pyplot(fig)
        else:
            st.warning("No recognized target column found in the dataset.")
        
        # Correlation heatmap
        st.subheader("Feature Correlations")
        
        # Preprocess before correlation
        df_processed = preprocess_data(df)
        
        if st.checkbox("Show correlation heatmap", value=True):
            try:
                corr_fig = plot_correlation_heatmap(df_processed)
                st.pyplot(corr_fig)
            except Exception as e:
                st.error(f"Error generating correlation heatmap: {e}")
        
        # Prepare for modeling
        if target_col:
            # Check if binary target exists
            binary_target = None
            if "Heart Attack Risk (Binary)" in df.columns:
                binary_target = "Heart Attack Risk (Binary)"
            elif "Heart Attack Risk" in df.columns and df["Heart Attack Risk"].nunique() == 2:
                binary_target = "Heart Attack Risk"
            
            if binary_target:
                # Store for modeling
                if "Heart Attack Risk (Text)" in df.columns:
                    X = df_processed.drop(columns=[binary_target, "Heart Attack Risk (Text)"])
                else:
                    X = df_processed.drop(columns=[binary_target])
                    
                y = df_processed[binary_target]
                
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.feature_names = X.columns
                
                st.success("Data is ready for modeling! Navigate to 'Model Training' to continue.")
            else:
                st.warning("Could not identify binary target variable for modeling.")
        else:
            st.warning("Target variable not found. Please ensure your dataset contains a heart attack risk column.")

# Model Training page
elif app_mode == "Model Training":
    st.markdown("<h2 class='sub-header'>Model Training and Evaluation</h2>", unsafe_allow_html=True)
    
    if st.session_state.X is None or st.session_state.y is None:
        st.warning("Please complete data exploration first!")
    else:
        # Train models button
        if st.button("Train Models"):
            with st.spinner("Training models... This may take a few minutes."):
                try:
                    models, X_train, X_test, y_train, y_test, scaler, X_train_scaled, X_test_scaled = train_models(
                        st.session_state.X, st.session_state.y
                    )
                    
                    # Store in session state
                    st.session_state.models = models
                    st.session_state.scaler = scaler
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    st.session_state.X_train_scaled = X_train_scaled
                    st.session_state.X_test_scaled = X_test_scaled
                    
                    # Create SHAP explainer for XGBoost
                    st.session_state.explainer = shap.Explainer(models['XGBoost']['model'], X_train_scaled)
                    
                    st.success("Models trained successfully!")
                except Exception as e:
                    st.error(f"Error training models: {e}")
        
        # Show model results if available
        if st.session_state.models is not None:
            st.subheader("Model Evaluation")
            
            selected_model = st.selectbox(
                "Select model to view results:",
                list(st.session_state.models.keys())
            )
            
            model_data = st.session_state.models[selected_model]
            
            # Display metrics
            col1, col2 = st.columns(2)
            
            with col1:
                # Accuracy
                accuracy = accuracy_score(st.session_state.y_test, model_data['predictions'])
                st.metric("Accuracy", f"{accuracy:.4f}")
                
                # Classification report
                st.subheader("Classification Report")
                report = classification_report(st.session_state.y_test, model_data['predictions'], output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.write(report_df)
            
            with col2:
                # Confusion matrix
                st.subheader("Confusion Matrix")
                cm_fig = plot_confusion_matrix(st.session_state.y_test, model_data['predictions'], selected_model)
                st.pyplot(cm_fig)
            
            # Feature importance
            st.subheader("Feature Importance")
            importance_fig = plot_feature_importance(
                model_data['model'], 
                st.session_state.feature_names,
                selected_model
            )
            st.pyplot(importance_fig)
            
            # Clinical context analysis
            st.subheader("Clinical Context Analysis")
            
            clinical_features, clinical_fig1, clinical_fig2, actionability_score = clinical_feature_analysis(
                model_data['model'],
                st.session_state.feature_names
            )
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.pyplot(clinical_fig1)
            
            with col2:
                st.pyplot(clinical_fig2)
                st.metric("Actionability Score", f"{actionability_score:.2f}", 
                          help="Higher score means more potential for risk reduction through lifestyle interventions")
            
            # Top modifiable risk factors
            st.subheader("Top Modifiable Risk Factors")
            top_modifiable = clinical_features[clinical_features['Category'] == 'Modifiable'].head(5)
            
            for i, row in top_modifiable.iterrows():
                st.write(f"- **{row['Feature']}** (Importance: {row['Importance']:.4f})")
            
            st.markdown("""
            <div class='info-box'>
            <p><strong>What This Means:</strong> The above factors are modifiable through lifestyle changes and medical interventions. Focus on these for risk reduction.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Download model results
            st.subheader("Download Results")
            
            def get_table_download_link(df, filename, text):
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">{text}</a>'
                return href
            
            st.markdown(get_table_download_link(clinical_features, 
                                                f"{selected_model}_feature_importance", 
                                                "Download Feature Importance Data"), 
                       unsafe_allow_html=True)

# Risk Prediction page
elif app_mode == "Risk Prediction":
    st.markdown("<h2 class='sub-header'>Personalized Risk Prediction</h2>", unsafe_allow_html=True)
    
    if st.session_state.models is None:
        st.warning("Please train models first!")
    else:
        st.markdown("""
        Enter patient information below to get a personalized heart attack risk prediction.
        This tool can help healthcare providers identify high-risk individuals who may benefit from preventive interventions.
        """)
        
        # Create two columns for input
        col1, col2 = st.columns(2)
        
        # Left column inputs
        with col1:
            st.subheader("Patient Demographics")
            age = st.slider("Age", 18, 100, 50)
            sex = st.radio("Sex", ["Male", "Female"])
            family_history = st.checkbox("Family History of Heart Disease")
            
            st.subheader("Lifestyle Factors")
            smoking = st.radio("Smoking Status", ["Non-smoker", "Former smoker", "Current smoker"])
            physical_activity = st.slider("Physical Activity (days per week)", 0, 7, 3)
            sleep_hours = st.slider("Average Sleep Hours", 1, 12, 7)
            stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
            
        # Right column inputs
        with col2:
            st.subheader("Clinical Measurements")
            cholesterol = st.slider("Cholesterol (mg/dL)", 100, 350, 200)
            blood_pressure = st.slider("Systolic Blood Pressure (mmHg)", 90, 200, 120)
            bmi = st.slider("BMI", 15.0, 45.0, 25.0, 0.1)
            heart_rate = st.slider("Resting Heart Rate (bpm)", 40, 120, 70)
            
            st.subheader("Medical Conditions")
            diabetes = st.checkbox("Diabetes")
            previous_heart_problems = st.checkbox("Previous Heart Problems")
            medication = st.checkbox("Taking Heart Medication")
        
        # Create input data dictionary
        input_data = {
            'Age': age,
            'Sex_M': 1 if sex == "Male" else 0,
            'Cholesterol': cholesterol,
            'Blood Pressure': blood_pressure,
            'Heart Rate': heart_rate,
            'BMI': bmi,
            'Family History': 1 if family_history else 0,
            'Smoking': 0 if smoking == "Non-smoker" else (1 if smoking == "Former smoker" else 2),
            'Physical Activity': physical_activity,
            'Sleep Hours': sleep_hours,
            'Stress Level': stress_level,
            'Diabetes': 1 if diabetes else 0,
            'Previous Heart Problems': 1 if previous_heart_problems else 0,
            'Medication Use': 1 if medication else 0
        }
        
        # Model selection for prediction
        selected_model = st.selectbox(
            "Select model for prediction:",
            list(st.session_state.models.keys())
        )
        
        # Make prediction
        if st.button("Predict Risk"):
            with st.spinner("Calculating risk..."):
                try:
                    model = st.session_state.models[selected_model]['model']
                    scaler = st.session_state.scaler
                    
                    # Get explainer if available
                    explainer = st.session_state.explainer if selected_model == "XGBoost" else None
                    
                    # Make prediction
                    prediction, probability, shap_values = predict_individual_risk(
                        input_data,
                        model,
                        st.session_state.X.columns,
                        scaler,
                        explainer
                    )
                    
                    # Display results
                    st.subheader("Prediction Results")
                    
                    # Risk level
                    risk_level = "High" if prediction == 1 else "Low"
                    risk_color = "#FF4B4B" if risk_level == "High" else "#2E7D32"
                    
                    st.markdown(f"""
                    <div style="padding: 20px; border-radius: 10px; background-color: {risk_color}20; 
                                margin-bottom: 20px; border-left: 5px solid {risk_color};">
                        <h3 style="color: {risk_color};">Risk Level: {risk_level}</h3>
                        <h4>Probability of Heart Attack Risk: {probability:.1%}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display SHAP values if available
                    if shap_values is not None:
                        st.subheader("Risk Factor Analysis")
                        st.write("The chart below shows how each factor contributes to the prediction:")
                        
                        # Using matplotlib to display SHAP values
                        fig, ax = plt.subplots(figsize=(10, 6))
                        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                        st.pyplot(fig)
                        
                        # Actionable insights
                        st.subheader("Personalized Recommendations")
                        
                        # Get most influential modifiable factors
                        feature_values = pd.Series(shap_values[0].values, index=st.session_state.X.columns)
                        
                        # Define modifiable factors
                        modifiable_factors = [
                            'Cholesterol', 'Blood Pressure', 'BMI', 'Smoking', 
                            'Physical Activity', 'Sleep Hours', 'Stress Level'
                        ]
                        
                        # Find modifiable factors that contribute to risk
                        risky_factors = []
                        for factor in modifiable_factors:
                            matching_columns = [col for col in feature_values.index if factor in col]
                            for col in matching_columns:
                                if feature_values[col] > 0:  # Contributes to higher risk
                                    risky_factors.append((col, feature_values[col]))
                        
                        # Sort by importance
                        risky_factors.sort(key=lambda x: x[1], reverse=True)
                        
                        if risky_factors:
                            st.markdown("<h4 style='color: #1E88E5;'>Focus Areas for Risk Reduction:</h4>", unsafe_allow_html=True)
                            for factor, value in risky_factors[:3]:
                                st.markdown(f"- **{factor}**: Significant impact on risk prediction")
                            
                            # Generic recommendations
                            st.markdown("""
                            ### General Recommendations:
                            - Consult with a healthcare provider for personalized medical advice
                            - Consider regular cardiovascular screenings
                            - Maintain a heart-healthy diet and regular physical activity
                            - Follow medication regimens as prescribed by your doctor
                            """)
                        else:
                            st.write("No significant modifiable risk factors identified.")
                    
                    # Disclaimer
                    st.markdown("""
                    <div style="font-size: 0.8rem; padding: 10px; background-color: #F8F9FA; border-radius: 5px; margin-top: 20px;">
                    <strong>Disclaimer:</strong> This prediction is for informational purposes only and does not constitute medical advice.
                    Always consult with qualified healthcare providers for diagnosis and treatment decisions.
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error making prediction: {e}")

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 40px; padding: 20px; background-color: #F0F2F6; border-radius: 5px;">
<p>Heart Attack Risk Prediction App | Created with ❤️ and Streamlit</p>
</div>
""", unsafe_allow_html=True)