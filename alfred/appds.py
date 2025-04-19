import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from imblearn.combine import SMOTETomek
import base64
from io import BytesIO
import time

# Check for optional dependencies
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

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
        margin: 10px 0;
    }
    .metric-box {
        background-color: #FFFFFF;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 5px 0;
    }
    .high-risk {
        background-color: #FF5252;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .low-risk {
        background-color: #4CAF50;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
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
    if 'target_col' not in st.session_state:
        st.session_state.target_col = None

initialize_session_state()

# Helper functions
def generate_sample_data():
    """Generate synthetic heart disease data for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    data = {
        'Age': np.random.randint(20, 80, size=n_samples),
        'Sex': np.random.choice(['Male', 'Female'], size=n_samples, p=[0.55, 0.45]),
        'ChestPainType': np.random.choice(['Typical', 'Atypical', 'Non-anginal', 'Asymptomatic'], size=n_samples),
        'RestingBP': np.random.randint(90, 200, size=n_samples),
        'Cholesterol': np.random.randint(120, 400, size=n_samples),
        'FastingBS': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),
        'RestingECG': np.random.choice(['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'], size=n_samples),
        'MaxHR': np.random.randint(60, 200, size=n_samples),
        'ExerciseAngina': np.random.choice(['No', 'Yes'], size=n_samples),
        'Oldpeak': np.round(np.random.uniform(0, 6, size=n_samples), 1),
        'ST_Slope': np.random.choice(['Up', 'Flat', 'Down'], size=n_samples),
        'HeartDisease': np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    }
    return pd.DataFrame(data)

@st.cache_data
def load_sample_data():
    """Load sample data with fallback to generated data"""
    try:
        df = pd.read_csv("heart.csv")
        return df
    except FileNotFoundError:
        st.warning("Sample dataset not found. Using synthetic data for demonstration.")
        return generate_sample_data()

def preprocess_data(df, target_col):
    """Preprocess the data for modeling"""
    # Handle missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())
    
    # Convert categorical variables to dummies
    df_processed = pd.get_dummies(df, drop_first=True)
    
    # Ensure target column exists
    if target_col not in df_processed.columns:
        st.error(f"Target column '{target_col}' not found in processed data")
        return None, None
    
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]
    
    return X, y

def plot_correlation_heatmap(df):
    """Plot correlation heatmap for numeric features"""
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if len(numeric_df.columns) < 2:
        st.warning("Not enough numeric columns for correlation heatmap")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    corr = numeric_df.corr()
    mask = np.triu(corr)
    sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', ax=ax, vmin=-1, vmax=1)
    plt.title("Feature Correlation Heatmap")
    return fig

def train_models(X, y):
    """Train multiple classification models"""
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
    training_times = {}
    
    model_configs = {
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    for model_name, model in model_configs.items():
        with st.spinner(f'Training {model_name} model...'):
            try:
                start_time = time.time()
                model.fit(X_train_scaled, y_train)
                training_time = time.time() - start_time
                training_times[model_name] = training_time
                
                models[model_name] = {
                    'model': model,
                    'predictions': model.predict(X_test_scaled),
                    'probabilities': model.predict_proba(X_test_scaled)[:, 1],
                    'training_time': training_time
                }
            except Exception as e:
                st.error(f"Error training {model_name}: {e}")
    
    return models, X_train, X_test, y_train, y_test, scaler, X_train_scaled, X_test_scaled, training_times

def plot_confusion_matrix(y_true, y_pred, title):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {title}")
    return fig

def plot_roc_curve(y_true, probas, models):
    """Plot ROC curve for multiple models"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for model_name, y_prob in probas.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    
    return fig

def plot_precision_recall_curve(y_true, probas, models):
    """Plot precision-recall curve for multiple models"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for model_name, y_prob in probas.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ax.plot(recall, precision, lw=2, label=f'{model_name}')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="best")
    
    return fig

def plot_feature_importance(model, feature_names, title):
    """Plot feature importance for tree-based models or coefficients for linear models"""
    try:
        if hasattr(model, 'feature_importances_'):
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
        
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
            indices = np.argsort(importances)[::-1]
            features = np.array(feature_names)[indices]
            values = importances[indices]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=values[:15], y=features[:15], palette='viridis', ax=ax)
            plt.title(f'{title} Feature Coefficients (Absolute Values)')
            plt.xlabel("Coefficient Magnitude")
            plt.ylabel("Feature")
            return fig
        
        else:
            st.warning(f"Feature importance not available for {title} model.")
            return None
            
    except Exception as e:
        st.error(f"Error generating feature importance: {e}")
        return None

def clinical_feature_analysis(model, feature_names):
    """Analyze features from a clinical perspective"""
    try:
        # Get feature importances based on model type
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])  # For linear models
        else:
            # Create mock importances if model doesn't support them
            importances = np.ones(len(feature_names)) / len(feature_names)
        
        # Define modifiability categories
        modifiable = [
            'Cholesterol', 'BloodPressure', 'BMI', 'Exercise', 
            'Smoking', 'Alcohol', 'PhysicalActivity', 'Stress',
            'Sedentary', 'Sleep'
        ]
        
        semi_modifiable = [
            'Medication', 'Obesity', 'HeartRate', 'Triglycerides'
        ]
        
        non_modifiable = [
            'Age', 'Sex', 'FamilyHistory', 'PreviousHeartProblems', 
            'Diabetes'
        ]
        
        # Categorize each feature
        feature_categories = []
        for feature in feature_names:
            feature_lower = feature.lower()
            if any(factor.lower() in feature_lower for factor in modifiable):
                category = 'Modifiable'
            elif any(factor.lower() in feature_lower for factor in semi_modifiable):
                category = 'Semi-Modifiable'
            elif any(factor.lower() in feature_lower for factor in non_modifiable):
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
        sns.barplot(x='Importance', y='Feature', hue='Category', 
                    data=feature_df.head(15), ax=ax1)
        plt.title('Top 15 Feature Importances with Clinical Context')
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
        ax2.set_title('Risk Factor Modifiability Distribution')
        ax2.axis('equal')
        
        # Create actionability score
        modifiable_importance = category_importance[
            category_importance['Category'] == 'Modifiable'
        ]['Importance'].sum()
        
        semi_modifiable_importance = category_importance[
            category_importance['Category'] == 'Semi-Modifiable'
        ]['Importance'].sum()
        
        total_importance = category_importance['Importance'].sum()
        
        actionability_score = (
            modifiable_importance + 0.5 * semi_modifiable_importance
        ) / total_importance
        
        return feature_df, fig1, fig2, actionability_score
        
    except Exception as e:
        st.error(f"Error in clinical feature analysis: {e}")
        return None, None, None, None

def predict_individual_risk(input_data, model, X_columns, scaler, explainer=None):
    """Make a prediction for an individual"""
    try:
        df_input = pd.DataFrame([input_data])
        
        # Convert categorical variables to dummies
        df_input_dummies = pd.get_dummies(df_input)
        
        # Ensure all expected columns are present
        missing_cols = set(X_columns) - set(df_input_dummies.columns)
        for col in missing_cols:
            df_input_dummies[col] = 0
            
        # Reorder columns to match training data
        df_input_dummies = df_input_dummies[X_columns]
        
        # Scale the data
        df_input_scaled = scaler.transform(df_input_dummies)
        
        # Make prediction
        prediction = model.predict(df_input_scaled)[0]
        proba = model.predict_proba(df_input_scaled)[0][1]
        
        # Create SHAP explanation if available
        shap_values = None
        if explainer is not None and SHAP_AVAILABLE:
            try:
                shap_values = explainer(df_input_scaled)
            except Exception as e:
                st.warning(f"Could not generate SHAP explanation: {e}")
        
        return prediction, proba, shap_values
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None, None

def create_risk_gauge(probability):
    """Create a visual risk gauge"""
    fig, ax = plt.subplots(figsize=(6, 1))
    plt.axis('off')
    
    # Create colored background
    background = np.zeros((1, 100, 3))
    for i in range(100):
        if i < 25:
            background[0, i] = [0, 1, 0]  # Green
        elif i < 50:
            background[0, i] = [1, 1, 0]  # Yellow
        elif i < 75:
            background[0, i] = [1, 0.5, 0]  # Orange
        else:
            background[0, i] = [1, 0, 0]  # Red
    
    plt.imshow(background)
    
    # Add marker for probability
    marker_pos = int(probability * 100)
    plt.plot(marker_pos, 0, 'v', color='black', markersize=12)
    
    # Add percentage text
    plt.text(50, 0, f"{probability*100:.1f}%", 
            fontsize=12, fontweight='bold', 
            ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.8))
    
    return fig

def get_risk_level(probability):
    """Get risk level based on probability"""
    if probability < 0.25:
        return "Low Risk", "Continue with healthy lifestyle."
    elif probability < 0.5:
        return "Moderate Risk", "Consider lifestyle improvements and regular check-ups."
    elif probability < 0.75:
        return "High Risk", "Consult with healthcare provider soon."
    else:
        return "Very High Risk", "Urgent medical consultation recommended!"

def get_table_download_link(df, filename, text):
    """Generate a download link for a DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">{text}</a>'
    return href

# App pages
def home_page():
    """Home page with data loading functionality"""
    st.markdown("<h1 class='main-header'>❤️ Heart Attack Risk Prediction</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    This app helps predict the risk of heart attack using machine learning models.
    Upload your data or use our sample dataset to get started.
    """)
    
    # Data loading section
    st.markdown("<h3 class='sub-header'>Load Data</h3>", unsafe_allow_html=True)
    
    data_source = st.radio(
        "Choose data source:",
        ("Upload your own data", "Use sample data")
    )
    
    if data_source == "Upload your own data":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.data = df
                st.success("Data successfully loaded!")
            except Exception as e:
                st.error(f"Error loading file: {e}")
    else:
        if st.button("Load Sample Data"):
            sample_data = load_sample_data()
            st.session_state.data = sample_data
            st.success("Sample data loaded!")
    
    if st.session_state.data is not None:
        st.markdown("<p class='info-box'>✅ Data loaded! Navigate to 'Data Exploration' to continue.</p>", 
                   unsafe_allow_html=True)
        
        # Let user select target column
        st.subheader("Select Target Column")
        target_col = st.selectbox(
            "Choose the column representing heart attack risk:",
            st.session_state.data.columns
        )
        st.session_state.target_col = target_col

def data_exploration_page():
    """Data exploration and visualization page"""
    st.markdown("<h2 class='sub-header'>Data Exploration</h2>", unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("Please load data on the Home page first!")
        return
    
    df = st.session_state.data
    
    # Data overview
    st.subheader("Dataset Overview")
    st.write(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.checkbox("Show raw data", value=True):
            st.write(df.head())
    with col2:
        if st.checkbox("Show column information"):
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
    
    if st.session_state.target_col:
        fig, ax = plt.subplots(figsize=(8, 6))
        value_counts = df[st.session_state.target_col].value_counts()
        ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        plt.title(f"Distribution of {st.session_state.target_col}")
        st.pyplot(fig)
    else:
        st.warning("No target column selected.")
    
    # Correlation heatmap
    st.subheader("Feature Correlations")
    
    if st.checkbox("Show correlation heatmap", value=True):
        try:
            corr_fig = plot_correlation_heatmap(df)
            if corr_fig:
                st.pyplot(corr_fig)
        except Exception as e:
            st.error(f"Error generating correlation heatmap: {e}")
    
    # Prepare for modeling
    if st.session_state.target_col:
        try:
            X, y = preprocess_data(df, st.session_state.target_col)
            if X is not None and y is not None:
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.feature_names = X.columns
                
                st.success("Data is ready for modeling! Navigate to 'Model Training' to continue.")
        except Exception as e:
            st.error(f"Error preparing data for modeling: {e}")

def model_training_page():
    """Model training and evaluation page"""
    st.markdown("<h2 class='sub-header'>Model Training and Evaluation</h2>", unsafe_allow_html=True)
    
    if st.session_state.X is None or st.session_state.y is None:
        st.warning("Please complete data exploration first!")
        return
    
    # Train models button
    if st.button("Train Models"):
        with st.spinner("Training models... This may take a few minutes."):
            try:
                start_time = time.time()
                (models, X_train, X_test, y_train, y_test, 
                 scaler, X_train_scaled, X_test_scaled, training_times) = train_models(
                    st.session_state.X, st.session_state.y
                )
                
                # Store in session state
                st.session_state.models = models
                st.session_state.scaler = scaler
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.X_train_scaled = X_train_scaled
                st.session_state.X_test_scaled = X_test_scaled
                st.session_state.training_times = training_times
                
                # Create SHAP explainer for XGBoost if available
                if SHAP_AVAILABLE and 'XGBoost' in models:
                    try:
                        st.session_state.explainer = shap.Explainer(
                            models['XGBoost']['model'], 
                            X_train_scaled
                        )
                    except Exception as e:
                        st.warning(f"Could not initialize SHAP explainer: {e}")
                
                st.success(f"Models trained successfully in {time.time() - start_time:.2f} seconds!")
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
            report = classification_report(
                st.session_state.y_test, 
                model_data['predictions'], 
                output_dict=True
            )
            report_df = pd.DataFrame(report).transpose()
            st.write(report_df)
        
        with col2:
            # Confusion matrix
            st.subheader("Confusion Matrix")
            cm_fig = plot_confusion_matrix(
                st.session_state.y_test, 
                model_data['predictions'], 
                selected_model
            )
            st.pyplot(cm_fig)
        
        # Feature importance
        st.subheader("Feature Importance")
        importance_fig = plot_feature_importance(
            model_data['model'], 
            st.session_state.feature_names,
            selected_model
        )
        
        if importance_fig:
            st.pyplot(importance_fig)
        else:
            st.info(f"Feature importance visualization not available for {selected_model}.")
        
        # Clinical context analysis
        st.subheader("Clinical Context Analysis")
        
        clinical_features, clinical_fig1, clinical_fig2, actionability_score = clinical_feature_analysis(
            model_data['model'],
            st.session_state.feature_names
        )
        
        if clinical_fig1 and clinical_fig2:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.pyplot(clinical_fig1)
            
            with col2:
                st.pyplot(clinical_fig2)
                st.metric("Actionability Score", f"{actionability_score:.2f}", 
                          help="Higher score means more potential for risk reduction through interventions")
            
            # Top modifiable risk factors
            st.subheader("Top Modifiable Risk Factors")
            if clinical_features is not None:
                top_modifiable = clinical_features[
                    clinical_features['Category'] == 'Modifiable'
                ].head(5)
                
                for i, row in top_modifiable.iterrows():
                    st.write(f"- **{row['Feature']}** (Importance: {row['Importance']:.4f})")
                
                st.markdown("""
                <div class='info-box'>
                <p><strong>Clinical Insight:</strong> The above factors are modifiable through lifestyle changes 
                and medical interventions. Focus on these for risk reduction.</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Download model results
        st.subheader("Download Results")
        
        if clinical_features is not None:
            st.markdown(get_table_download_link(
                clinical_features, 
                f"{selected_model}_feature_importance", 
                "Download Feature Importance Data"
            ), unsafe_allow_html=True)

def model_comparison_page():
    """Model comparison page"""
    st.markdown("<h2 class='sub-header'>Model Comparison</h2>", unsafe_allow_html=True)
    
    if st.session_state.models is None:
        st.warning("Please train models first!")
        return
    
    st.write("Compare the performance of different models to select the best one for heart attack risk prediction.")
    
    # Metrics comparison
    st.subheader("Performance Metrics Comparison")
    
    metrics = []
    for model_name, model_data in st.session_state.models.items():
        y_true = st.session_state.y_test
        y_pred = model_data['predictions']
        y_prob = model_data['probabilities']
        
        # Calculate metrics
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        metrics.append({
            'Model': model_name,
            'Accuracy': acc,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'Precision': precision,
            'F1 Score': f1,
            'AUC': roc_auc,
            'Training Time (s)': model_data['training_time']
        })
    
    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.set_index('Model')
    
    # Format the values
    formatted_df = metrics_df.style.format("{:.4f}").highlight_max(axis=0, color='#AED6F1')
    st.dataframe(formatted_df)
    
    # ROC Curve Comparison
    st.subheader("ROC Curve Comparison")
    
    # Prepare probabilities dict for plotting
    probas = {model_name: model_data['probabilities'] 
              for model_name, model_data in st.session_state.models.items()}
    
    # Plot ROC curves
    roc_fig = plot_roc_curve(st.session_state.y_test, probas, st.session_state.models)
    st.pyplot(roc_fig)
    
    # Precision-Recall Curve
    st.subheader("Precision-Recall Curve Comparison")
    pr_fig = plot_precision_recall_curve(st.session_state.y_test, probas, st.session_state.models)
    st.pyplot(pr_fig)
    
    # Training time comparison
    st.subheader("Training Time Comparison")
    times_df = pd.DataFrame({
        'Model': list(st.session_state.training_times.keys()),
        'Time (s)': list(st.session_state.training_times.values())
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Time (s)', y='Model', data=times_df, palette='viridis', ax=ax)
    plt.title('Model Training Time Comparison')
    plt.xlabel('Time (seconds)')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Model selection recommendation
    st.subheader("Model Recommendation")
    
    # Find the best model based on AUC
    best_model_name = metrics_df['AUC'].idxmax()
    best_auc = metrics_df.loc[best_model_name, 'AUC']
    
    st.markdown(f"""
    <div class='info-box'>
    <p>Based on the comparison metrics, the recommended model is:</p>
    <h3 style='color: #1E88E5;'>{best_model_name}</h3>
    <p>This model achieves the highest AUC score of {float(best_auc):.4f}, indicating strong discriminative ability 
    between heart attack risk classes.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # SHAP Summary Plot for best model
    if SHAP_AVAILABLE and best_model_name == 'XGBoost' and st.session_state.explainer is not None:
        st.subheader(f"SHAP Feature Impact for {best_model_name}")
        
        if st.button("Generate SHAP Analysis (may take a minute)"):
            with st.spinner("Generating SHAP values..."):
                try:
                    # Use a sample of the test data for computation efficiency
                    sample_size = min(100, len(st.session_state.X_test_scaled))
                    X_sample = st.session_state.X_test_scaled[:sample_size]
                    
                    # Create and display SHAP summary plot
                    fig, ax = plt.subplots(figsize=(10, 8))
                    shap.summary_plot(
                        st.session_state.explainer(X_sample), 
                        X_sample, 
                        feature_names=st.session_state.feature_names, 
                        show=False
                    )
                    plt.title(f"SHAP Feature Impact for {best_model_name}")
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.markdown("""
                    <div class='info-box'>
                    <p><strong>How to interpret:</strong> Features are ordered by their global importance. 
                    Red points indicate higher feature values, while blue points indicate lower feature values. 
                    Points further to the right have a higher positive impact on risk prediction.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error generating SHAP plot: {e}")
    elif not SHAP_AVAILABLE:
        st.info("SHAP is not installed. Install with 'pip install shap' for feature impact analysis.")

def risk_prediction_page():
    """Personalized risk prediction page"""
    st.markdown("<h2 class='sub-header'>Personalized Heart Attack Risk Prediction</h2>", unsafe_allow_html=True)
    
    if st.session_state.models is None or st.session_state.scaler is None:
        st.warning("Please train models first!")
        return
    
    st.write("Enter patient information to get a personalized heart attack risk prediction.")
    
    # Select model for prediction
    model_name = st.selectbox(
        "Select model for prediction:",
        list(st.session_state.models.keys()),
        index=list(st.session_state.models.keys()).index("XGBoost") if "XGBoost" in st.session_state.models else 0
    )
    
    selected_model = st.session_state.models[model_name]['model']
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Form Input", "CSV Upload"])
    
    with tab1:
        st.subheader("Enter Patient Information")
        
        # Create columns for better layout
        col1, col2, col3 = st.columns(3)
        
        # Create dictionary to hold input values
        input_data = {}
        
        # Demographic information
        with col1:
            st.markdown("##### Demographics")
            input_data['Age'] = st.number_input("Age", min_value=18, max_value=120, value=50)
            input_data['Sex'] = st.selectbox("Sex", ["Male", "Female"])
            input_data['ChestPainType'] = st.selectbox(
                "Chest Pain Type", 
                ["Typical", "Atypical", "Non-anginal", "Asymptomatic"]
            )
            
        # Health metrics
        with col2:
            st.markdown("##### Health Metrics")
            input_data