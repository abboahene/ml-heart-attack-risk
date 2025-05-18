# -*- coding: utf-8 -*-
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
from imblearn.combine import SMOTETomek # You might need: pip install imbalanced-learn
import shap # You might need: pip install shap
import base64
from io import BytesIO
import time
import warnings
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import make_pipeline

# Suppress warnings
warnings.filterwarnings('ignore')

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
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .info-box {
        background-color: #F0F2F6;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
        border: 1px solid #E0E0E0;
    }
    .model-metrics {
        display: flex;
        justify-content: space-between;
    }
    .shap-value {
        font-weight: bold;
        color: #FF4B4B;
    }
    /* Ensure plots have enough space */
    .stPlotlyChart, .stImage {
        margin-bottom: 20px;
    }
    /* Make buttons stand out more */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    /* Style for risk level display */
    .risk-high {
        background-color:#FF5252;
        padding:20px;
        border-radius:10px;
        text-align:center;
    }
    .risk-low {
        background-color:#4CAF50;
        padding:20px;
        border-radius:10px;
        text-align:center;
    }
    .risk-text {
        color:white;
        font-size: 1.8rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Main header
st.markdown("<h1 class='main-header'>❤️ Heart Attack Risk Prediction</h1>", unsafe_allow_html=True)

# App description
st.markdown("""
This app helps predict the risk of heart attack using machine learning models.
Upload your data or use our sample dataset to get started. Navigate using the sidebar.
""")

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/heart-health.png", width=100)
st.sidebar.title("Controls")

# Functions for data preprocessing and modeling
@st.cache_data
def load_sample_data():
    """Generates or loads sample heart attack risk data."""
    try:
        # Sample data with realistic heart attack risk features
        data = {
            'Age': np.random.randint(30, 80, 1000),
            'Sex': np.random.choice(['Male', 'Female'], 1000),
            'Cholesterol': np.random.randint(150, 350, 1000), # Increased max range slightly
            'Blood_Pressure': np.random.randint(90, 180, 1000), # Systolic BP
            'Heart_Rate': np.random.randint(60, 105, 1000), # Increased max range slightly
            'BMI': np.round(np.random.uniform(18.5, 40, 1000), 1), # Wider BMI range
            'Smoking': np.random.choice(['Never', 'Former', 'Current'], 1000),
            'Diabetes': np.random.choice(['No', 'Yes'], 1000),
            'Family_History': np.random.choice(['No', 'Yes'], 1000),
            'Exercise_Hours': np.round(np.random.uniform(0, 15, 1000), 1), # Increased max range
            'Stress_Level': np.random.randint(1, 10, 1000),
            'Heart_Attack_Risk': np.random.choice([0, 1], 1000, p=[0.65, 0.35]) # Slightly higher risk prevalence
        }
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        st.error(f"Error generating sample data: {e}")
        return None

# @st.cache_data # Caching this might be problematic if NaNs change
def preprocess_data(df, fit_scaler=False, scaler=None, training_columns=None):
    """
    Preprocesses the data: handles missing values, encodes categoricals.
    Optionally fits a scaler or uses a pre-fitted one.
    Aligns columns with training data if provided.
    """
    df_processed = df.copy() # Avoid modifying original df

    # Handle missing values
    for col in df_processed.columns:
        if df_processed[col].isnull().any(): # Only process if NaNs exist
            if df_processed[col].dtype == 'object':
                # Fill with mode, handle potential empty mode list
                mode_val = df_processed[col].mode()
                if not mode_val.empty:
                    df_processed[col] = df_processed[col].fillna(mode_val[0])
                else:
                     df_processed[col] = df_processed[col].fillna('Unknown') # Fallback
            elif pd.api.types.is_numeric_dtype(df_processed[col]):
                 # Fill numeric with median
                median_val = df_processed[col].median()
                df_processed[col] = df_processed[col].fillna(median_val)
            # Add handling for other types if necessary

    # Convert categorical variables to dummies
    # Use dummy_na=False to avoid columns for NaN categories
    df_processed = pd.get_dummies(df_processed, drop_first=True, dummy_na=False)

    # Align columns with training data if training_columns are provided (for prediction)
    if training_columns is not None:
        # Get missing columns in the new data
        missing_cols = set(training_columns) - set(df_processed.columns)
        for c in missing_cols:
            df_processed[c] = 0 # Add missing columns with 0
        # Ensure the order of columns is the same as in training
        df_processed = df_processed[training_columns]

    # Separate features (X) and target (y) if target exists
    target_col = None
    possible_targets = ['Heart_Attack_Risk', 'HeartAttackRisk', 'Risk', 'target', 'output']
    for col in possible_targets:
        if col in df_processed.columns:
            target_col = col
            break

    if target_col:
        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]
    else:
        X = df_processed # Assume all columns are features if no target found
        y = None

    # Scaling
    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X, y, scaler, X.columns.tolist() # Return fitted scaler and column names
    elif scaler is not None:
        X_scaled = scaler.transform(X)
        return X_scaled, X.columns.tolist() # Return scaled data and column names
    else:
        # If no scaling requested, return X and y
        return X, y, None, X.columns.tolist() if X is not None else []


def plot_correlation_heatmap(df):
    """Plots the correlation heatmap for numerical features."""
    fig, ax = plt.subplots(figsize=(14, 10))
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=np.number)
    if not numeric_df.empty:
        corr = numeric_df.corr()
        mask = np.triu(corr)
        sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', ax=ax, vmin=-1, vmax=1,
                    cbar_kws={"shrink": .8}) # Adjust color bar size
        plt.title("Feature Correlation Heatmap (Numeric Features)", fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
    else:
        ax.text(0.5, 0.5, "No numeric columns found for correlation.", ha='center', va='center')
    return fig

from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

def train_models(X_train_df, y_train_series):
    """Trains multiple classification models after resampling and scaling, including an enhanced stacking classifier."""
    models = {}
    training_times = {}

    # Convert to numpy arrays
    X_np = X_train_df.values
    y_np = y_train_series.values

    st.write(f"Original training data shape: {X_np.shape}")
    st.write(f"Original training target distribution: {np.bincount(y_np)}")

    # Resampling
    smote_tomek = SMOTETomek(random_state=42, n_jobs=-1)
    try:
        X_resampled, y_resampled = smote_tomek.fit_resample(X_np, y_np)
        st.write(f"Resampled training data shape: {X_resampled.shape}")
        st.write(f"Resampled training target distribution: {np.bincount(y_resampled)}")
    except Exception as e:
        st.error(f"Error during resampling with SMOTETomek: {e}")
        st.warning("Proceeding without resampling.")
        X_resampled, y_resampled = X_np, y_np

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_resampled)

    # --- Base Models ---
    model_defs = {
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, solver='liblinear', class_weight='balanced'),
        'SVM': SVC(random_state=42, probability=True, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, early_stopping=True)
    }

    # --- Train Base Models ---
    for name, model_instance in model_defs.items():
        with st.spinner(f'Training {name} model...'):
            start_time = time.time()
            try:
                model_instance.fit(X_train_scaled, y_resampled)
                elapsed_time = time.time() - start_time
                training_times[name] = elapsed_time
                models[name] = {'model': model_instance}
                st.write(f"✔️ {name} trained in {elapsed_time:.2f} sec")
            except Exception as e:
                st.error(f"Error training {name}: {e}")

    # --- Stacking Classifier ---
    with st.spinner('Training Stacking Classifier...'):
        start_time = time.time()
        try:
            estimators = [
                ('lr', model_defs['Logistic Regression']),
                ('svm', model_defs['SVM']),
                ('rf', model_defs['Random Forest']),
                ('xgb', model_defs['XGBoost']),
                ('mlp', model_defs['Neural Network'])
            ]

            final_estimator = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

            stacking_clf = StackingClassifier(
                estimators=estimators,
                final_estimator=final_estimator,
                passthrough=True,  # Allows meta-model to see original features
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                n_jobs=-1
            )
            stacking_clf.fit(X_train_scaled, y_resampled)

            elapsed_time = time.time() - start_time
            training_times['Stacking Classifier'] = elapsed_time
            models['Stacking Classifier'] = {'model': stacking_clf}
            st.write(f"✔️ Stacking Classifier trained in {elapsed_time:.2f} sec")
        except Exception as e:
            st.error(f"Error training Stacking Classifier: {e}")

    return models, scaler, training_times, X_train_scaled, y_resampled

def evaluate_models(models, X_test_scaled, y_test):
    """Evaluates trained models on the test set."""
    results = {}
    for name, model_data in models.items():
        model = model_data['model']
        try:
            predictions = model.predict(X_test_scaled)
            probabilities = model.predict_proba(X_test_scaled)[:, 1]
            results[name] = {
                'predictions': predictions,
                'probabilities': probabilities
            }
        except Exception as e:
            st.error(f"Error evaluating {name}: {e}")
            results[name] = {'predictions': None, 'probabilities': None} # Indicate failure
    return results


def plot_confusion_matrix_func(y_true, y_pred, title):
    """Plots a confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'])
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("Actual Label", fontsize=12)
        plt.title(f"Confusion Matrix - {title}", fontsize=14)
        plt.tight_layout()
    except Exception as e:
        ax.text(0.5, 0.5, f"Error plotting CM: {e}", ha='center', va='center')
    return fig

def plot_roc_curve_func(y_true, model_results, title="ROC Curve Comparison"):
    """Plots ROC curves for multiple models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance') # Add chance line

    for model_name, data in model_results.items():
        if data.get('probabilities') is not None:
            try:
                fpr, tpr, _ = roc_curve(y_true, data['probabilities'])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
            except Exception as e:
                st.warning(f"Could not plot ROC for {model_name}: {e}")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=16)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig

def plot_precision_recall_curve_func(y_true, model_results, title="Precision-Recall Curve Comparison"):
    """Plots Precision-Recall curves for multiple models."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for model_name, data in model_results.items():
         if data.get('probabilities') is not None:
            try:
                precision, recall, _ = precision_recall_curve(y_true, data['probabilities'])
                # Calculate Average Precision (AP)
                ap = auc(recall, precision)
                ax.plot(recall, precision, lw=2, label=f'{model_name} (AP = {ap:.3f})')
            except Exception as e:
                st.warning(f"Could not plot P-R for {model_name}: {e}")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=16)
    ax.legend(loc="best", fontsize=10) # Changed location
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig

def plot_feature_importance(model, feature_names, title, top_n=20):
    """Plots feature importances or coefficients."""
    importances = None
    importance_label = "Importance"

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models, use absolute coefficients
        importances = np.abs(model.coef_[0])
        importance_label = "Coefficient Magnitude"
    else:
        st.info(f"Feature importance not directly available for {type(model).__name__}.")
        return None # Model doesn't support direct feature importance

    if importances is not None:
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=(12, max(6, len(feature_importance_df) * 0.4))) # Dynamic height
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis', ax=ax)
        plt.title(f'{title} Feature Importances/Coefficients', fontsize=16)
        plt.xlabel(importance_label, fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.tight_layout()
        return fig
    return None


def clinical_feature_analysis(model, feature_names):
    """Analyzes feature importance based on clinical modifiability."""
    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        # Handle models without direct importance (e.g., some SVM kernels)
        # Create mock importances or skip - let's assign equal importance for now
        st.warning(f"Using equal importance for clinical analysis as {type(model).__name__} lacks direct feature importance.")
        importances = np.ones(len(feature_names)) / len(feature_names)

    # Define modifiability categories based on common knowledge
    # Use lowercase feature names for matching
    feature_names_lower = [f.lower() for f in feature_names]
    input_feature_names_lower = {f.lower(): f for f in feature_names} # Map back to original case

    # Keywords for categorization (adjust as needed for specific feature names)
    modifiable_kws = ['cholesterol', 'blood_pressure', 'bmi', 'exercise', 'smoking', 'stress', 'sedentary']
    semi_modifiable_kws = ['medication', 'heart_rate'] # Heart rate can be influenced by fitness/meds
    non_modifiable_kws = ['age', 'sex', 'family_history', 'diabetes'] # Diabetes status itself often less modifiable than risk factors

    feature_categories = []
    original_case_features = []

    for feature_lower in feature_names_lower:
        category = 'Other' # Default category
        if any(kw in feature_lower for kw in modifiable_kws):
            category = 'Modifiable'
        elif any(kw in feature_lower for kw in semi_modifiable_kws):
            category = 'Semi-Modifiable'
        elif any(kw in feature_lower for kw in non_modifiable_kws):
            category = 'Non-Modifiable'
        feature_categories.append(category)
        original_case_features.append(input_feature_names_lower[feature_lower]) # Get original case feature name

    feature_df = pd.DataFrame({
        'Feature': original_case_features, # Use original case names
        'Importance': importances,
        'Category': feature_categories
    })
    feature_df = feature_df.sort_values('Importance', ascending=False)

    category_importance = feature_df.groupby('Category')['Importance'].sum().reset_index()
    total_importance = category_importance['Importance'].sum()
    # Handle division by zero if total importance is 0
    if total_importance == 0: total_importance = 1

    # --- Plot Feature Importance with Clinical Context ---
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    top_features = feature_df.head(15)
    sns.barplot(x='Importance', y='Feature', hue='Category', data=top_features, ax=ax1, dodge=False) # dodge=False stacks hues
    plt.title('Top 15 Feature Importances by Clinical Modifiability', fontsize=16)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.legend(title='Modifiability')
    plt.tight_layout()

    # --- Plot Categorical Pie Chart ---
    fig2, ax2 = plt.subplots(figsize=(8, 8)) # Made it square
    colors = {'Modifiable': '#2ecc71', 'Semi-Modifiable': '#f1c40f', 'Non-Modifiable': '#e74c3c', 'Other': '#95a5a6'}
    pie_colors = [colors.get(cat, '#bdc3c7') for cat in category_importance['Category']]
    ax2.pie(category_importance['Importance'],
            labels=category_importance['Category'],
            autopct='%1.1f%%',
            startangle=90,
            colors=pie_colors,
            wedgeprops={'edgecolor': 'white'}) # Add edge color
    ax2.set_title('Contribution to Prediction by Factor Modifiability', fontsize=16)
    ax2.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.

    # --- Calculate Actionability Score ---
    actionability_score = 0
    if 'Modifiable' in category_importance['Category'].values:
        actionability_score += category_importance.loc[category_importance['Category'] == 'Modifiable', 'Importance'].sum()
    if 'Semi-Modifiable' in category_importance['Category'].values:
        actionability_score += 0.5 * category_importance.loc[category_importance['Category'] == 'Semi-Modifiable', 'Importance'].sum()

    actionability_score /= total_importance

    return feature_df, fig1, fig2, actionability_score


def predict_individual_risk(input_data_dict, model, training_columns, scaler, explainer=None, feature_names_original_case=None):
    """Predicts risk for a single individual after preprocessing."""
    df_input = pd.DataFrame([input_data_dict])

    # Preprocess the input data using the fitted scaler and training columns
    # Handle missing values and dummies first (similar to preprocess_data)
    for col in df_input.columns:
        if df_input[col].isnull().any():
            if df_input[col].dtype == 'object':
                mode_val = df_input[col].mode()
                df_input[col] = df_input[col].fillna(mode_val[0] if not mode_val.empty else 'Unknown')
            elif pd.api.types.is_numeric_dtype(df_input[col]):
                median_val = df_input[col].median() # Or use mean/median from training data if stored
                df_input[col] = df_input[col].fillna(median_val)

    df_input_dummies = pd.get_dummies(df_input, drop_first=True, dummy_na=False)

    # Align columns with the training data columns
    df_input_reindexed = df_input_dummies.reindex(columns=training_columns, fill_value=0)

    # Scale the data using the *fitted* scaler
    try:
        df_input_scaled = scaler.transform(df_input_reindexed)
    except Exception as e:
        st.error(f"Error scaling input data: {e}")
        st.error(f"Input columns after reindexing: {df_input_reindexed.columns.tolist()}")
        st.error(f"Expected columns by scaler: {training_columns}")
        return None, None, None # Indicate error

    # Make prediction
    prediction = model.predict(df_input_scaled)[0]
    proba = model.predict_proba(df_input_scaled)[0][1] # Probability of class 1 (risk)

    # Create SHAP explanation if explainer exists
    shap_values_instance = None
    expected_value = None
    if explainer is not None:
        try:
            # SHAP expects numpy array usually
            shap_values_list = explainer.shap_values(df_input_scaled) # This might return a list for multi-class or just array for binary

            # Handle different SHAP explainer outputs
            if isinstance(shap_values_list, list) and len(shap_values_list) == 2:
                # Common for binary classification: values for class 0 and class 1
                shap_values_instance = shap_values_list[1][0] # Use values for class 1
            elif isinstance(shap_values_list, np.ndarray):
                 # Sometimes returns single array for binary
                 shap_values_instance = shap_values_list[0]
            else:
                st.warning(f"Unexpected SHAP values format: {type(shap_values_list)}. Cannot generate waterfall plot.")

            # Get expected value (base value for SHAP plots)
            if hasattr(explainer, 'expected_value'):
                 if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) == 2:
                     expected_value = explainer.expected_value[1] # Use expected value for class 1
                 elif isinstance(explainer.expected_value, (int, float)):
                     expected_value = explainer.expected_value
                 else:
                     st.warning(f"Unexpected format for explainer.expected_value: {type(explainer.expected_value)}")


        except Exception as e:
            st.error(f"Error generating SHAP explanation: {str(e)}")
            # Optionally provide more debug info:
            # st.error(f"Data shape sent to SHAP: {df_input_scaled.shape}")
            # st.error(f"Data type sent to SHAP: {df_input_scaled.dtype}")


    # Prepare SHAP Explanation object if values were generated
    shap_explanation = None
    if shap_values_instance is not None and expected_value is not None and feature_names_original_case is not None:
         # Ensure feature names match the number of SHAP values
         if len(feature_names_original_case) == len(shap_values_instance):
            shap_explanation = shap.Explanation(
                values=shap_values_instance,
                base_values=expected_value,
                data=df_input_scaled[0], # Use the scaled data instance
                feature_names=feature_names_original_case # Use original case feature names passed in
            )
         else:
             st.warning(f"Mismatch between number of feature names ({len(feature_names_original_case)}) and SHAP values ({len(shap_values_instance)}). Cannot create Explanation object.")


    return prediction, proba, shap_explanation # Return SHAP Explanation object

# App navigation
app_mode = st.sidebar.selectbox(
    "Choose a mode",
    ["Home", "Data Exploration", "Model Training", "Model Comparison", "Risk Prediction"]
)

# Initialize session state
default_session_state = {
    'data': None,
    'models': None,
    'X': None, # Store original X before preprocessing
    'y': None, # Store original y before preprocessing
    'X_train': None, 'X_test': None, 'y_train': None, 'y_test': None, # Split data
    'scaler': None,
    'explainer': None, # SHAP explainer (currently only for XGBoost)
    'feature_names': None, # Feature names *after* dummy encoding, used for training
    'training_columns': None, # Alias for feature_names, to be clear
    'training_times': None,
    'model_results': None, # Store predictions and probabilities
    'best_model_name': None,
    'X_train_scaled': None, # Needed for some SHAP explainers
    'X_test_scaled': None,
}

for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --- Page Implementations ---

# Home page
if app_mode == "Home":
    st.markdown("<h2 class='sub-header'>Welcome to the Heart Attack Risk Prediction App!</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1]) # Give more space to text

    with col1:
        st.markdown("""
        ### About this app
        This application utilizes machine learning to estimate the risk of heart attack based on user-provided health data or a sample dataset. It aims to provide insights into potential risk factors.

        ### Key features:
        - **Data Exploration:** Visualize and understand your dataset.
        - **Model Training:** Train various ML models (XGBoost, Random Forest, etc.) on your data.
        - **Model Comparison:** Evaluate and compare model performance using metrics like Accuracy, AUC, F1-score, and plots (ROC, Precision-Recall).
        - **Clinical Context:** Analyze feature importance in terms of modifiable vs. non-modifiable factors.
        - **Personalized Prediction:** Get risk estimates for individual profiles with explanations (using SHAP where available).
        - **Batch Prediction:** Predict risk for multiple individuals from a CSV file.

        ### How to use:
        1.  **Load Data:** Use the options below to upload your CSV or use the sample data.
        2.  **Explore (Optional):** Navigate to 'Data Exploration' via the sidebar to view data details.
        3.  **Train Models:** Go to 'Model Training' to train the predictive models.
        4.  **Compare (Optional):** Visit 'Model Comparison' to see how different models perform.
        5.  **Predict:** Use 'Risk Prediction' to assess risk for new individuals or batches.
        """, unsafe_allow_html=True) # Use unsafe_allow_html for markdown styling

    with col2:
        # Placeholder for an image or graphic
        st.image("https://img.icons8.com/external-flaticons-lineal-color-flat-icons/256/external-heart-anatomy-flaticons-lineal-color-flat-icons-3.png",
                 caption="Heart Health Analytics", use_column_width=True)
        st.info("Navigate through the app sections using the sidebar menu.")


    # Data loading section
    st.markdown("<h3 class='sub-header'>Load Data</h3>", unsafe_allow_html=True)

    data_source = st.radio(
        "Choose data source:",
        ("Use sample data", "Upload your own data"),
        key='data_source_radio' # Add key for state consistency
    )

    if data_source == "Upload your own data":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.data = df
                # Clear previous modeling results if new data is uploaded
                for key in ['models', 'scaler', 'X', 'y', 'X_train', 'X_test', 'y_train', 'y_test', 'feature_names', 'training_columns', 'model_results', 'explainer']:
                    st.session_state[key] = None
                st.success("Data successfully loaded! File details below.")
                st.write("Uploaded Data Head:")
                st.dataframe(df.head())
                st.write(f"Shape: {df.shape}")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                st.session_state.data = None # Ensure data state is None on error
    else:
        # Use sample data
        if st.button("Load Sample Data"):
            sample_data = load_sample_data()
            if sample_data is not None:
                st.session_state.data = sample_data
                 # Clear previous modeling results
                for key in ['models', 'scaler', 'X', 'y', 'X_train', 'X_test', 'y_train', 'y_test', 'feature_names', 'training_columns', 'model_results', 'explainer']:
                    st.session_state[key] = None
                st.success("Sample data loaded!")
                st.write("Sample Data Head:")
                st.dataframe(sample_data.head())
                st.write(f"Shape: {sample_data.shape}")
            else:
                st.error("Failed to load sample data.")

    if st.session_state.data is not None:
        st.markdown("<p class='info-box'>✅ Data loaded! Navigate to 'Data Exploration' or 'Model Training' using the sidebar.</p>", unsafe_allow_html=True)
    else:
        st.info("Please load data to proceed.")


# Data Exploration page
elif app_mode == "Data Exploration":
    st.markdown("<h2 class='sub-header'>Data Exploration</h2>", unsafe_allow_html=True)

    if st.session_state.data is None:
        st.warning("⚠️ Please load data on the Home page first!")
    else:
        df = st.session_state.data

        # Data overview
        st.subheader("Dataset Overview")
        st.write(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

        col1, col2 = st.columns(2)
        with col1:
            if st.checkbox("Show Raw Data Sample", value=False):
                st.dataframe(df.head(10)) # Show more rows
        with col2:
            if st.checkbox("Show Column Information (Data Types)", value=True):
                 st.dataframe(df.dtypes.reset_index().rename(columns={'index':'Column', 0:'DataType'}))

        # Missing values
        st.subheader("Missing Values Analysis")
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0] # Filter only columns with missing values
        if not missing_data.empty:
            st.write("Columns with missing values:")
            st.dataframe(missing_data.reset_index().rename(columns={'index':'Column', 0:'Missing Count'}))
            st.info("Missing values will be handled during preprocessing (median for numeric, mode for categoric).")
        else:
            st.success("✅ No missing values found in the dataset.")

        # Basic statistics for numeric columns
        st.subheader("Statistical Summary (Numeric Features)")
        numeric_cols = df.select_dtypes(include=np.number)
        if not numeric_cols.empty:
            st.dataframe(numeric_cols.describe().T) # Transpose for better readability
        else:
            st.write("No numeric columns found.")

        # Data distributions
        st.subheader("Data Distributions")
        numeric_cols_list = numeric_cols.columns.tolist()
        if numeric_cols_list:
            selected_col_hist = st.selectbox("Select numeric column for histogram:", numeric_cols_list)
            if selected_col_hist:
                fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
                sns.histplot(df[selected_col_hist].dropna(), kde=True, ax=ax_hist) # Drop NaNs for plotting
                plt.title(f"Distribution of {selected_col_hist}", fontsize=14)
                plt.xlabel(selected_col_hist, fontsize=12)
                plt.ylabel("Frequency", fontsize=12)
                plt.tight_layout()
                st.pyplot(fig_hist)
                plt.close(fig_hist) # Close the figure

        # Target variable distribution
        st.subheader("Target Variable Distribution")
        target_col = None
        possible_targets = ['Heart_Attack_Risk', 'HeartAttackRisk', 'Risk', 'target', 'output'] # Add 'output'
        for col in possible_targets:
            if col in df.columns:
                target_col = col
                st.success(f"✅ Target column identified as: '{target_col}'")
                break

        if target_col:
            if df[target_col].isnull().any():
                st.warning(f"Target column '{target_col}' contains missing values. Rows with missing target will be dropped.")
                df_cleaned = df.dropna(subset=[target_col])
            else:
                df_cleaned = df.copy()

            if df_cleaned[target_col].nunique() == 2:
                fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
                value_counts = df_cleaned[target_col].value_counts()
                ax_pie.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff9999'])
                ax_pie.set_title(f"Distribution of {target_col}", fontsize=14)
                st.pyplot(fig_pie)
                plt.close(fig_pie) # Close the figure

                # Prepare data for modeling (only if target is valid)
                st.markdown("<h3 class='sub-header'>Prepare Data for Modeling</h3>", unsafe_allow_html=True)
                st.write("Preprocessing data (handling missing values, encoding categoricals)...")
                try:
                    # Preprocess without scaling here, just get features and target
                    X_processed, y_processed, _, feature_names = preprocess_data(df_cleaned) # No scaling yet

                    if X_processed is not None and y_processed is not None:
                        st.session_state.X = X_processed # Store processed features (unscaled)
                        st.session_state.y = y_processed # Store processed target
                        st.session_state.feature_names = feature_names # Store feature names after dummies
                        st.session_state.training_columns = feature_names # Use same names for training consistency check

                        st.success("✅ Data successfully preprocessed and ready for modeling!")
                        st.write(f"Features shape: {st.session_state.X.shape}")
                        st.write(f"Target shape: {st.session_state.y.shape}")
                        st.write("First 5 rows of processed features (X):")
                        st.dataframe(st.session_state.X.head())
                        st.info("Navigate to 'Model Training' to proceed.")
                    else:
                         st.error("Error during preprocessing. Cannot proceed to modeling.")

                except Exception as e:
                    st.error(f"Error during preprocessing: {e}")

            else:
                st.error(f"❌ Target variable '{target_col}' is not binary (has {df[target_col].nunique()} unique values). Classification requires a binary target (0/1). Cannot proceed to modeling.")
                st.session_state.X = None # Reset states if target is invalid
                st.session_state.y = None
                st.session_state.feature_names = None
                st.session_state.training_columns = None
        else:
            st.warning("⚠️ No recognized target column found (looked for: 'Heart_Attack_Risk', 'HeartAttackRisk', 'Risk', 'target', 'output'). Please ensure your data has a binary target variable with one of these names to enable model training.")

        # Correlation heatmap (optional) - uses processed numeric features
        st.subheader("Feature Correlations (Numeric Only)")
        if st.checkbox("Show correlation heatmap", value=False):
            if st.session_state.X is not None: # Check if preprocessing was successful
                try:
                    # We need the numeric columns from the *processed* data
                    corr_fig = plot_correlation_heatmap(st.session_state.X)
                    st.pyplot(corr_fig)
                    plt.close(corr_fig) # Close the figure
                except Exception as e:
                    st.error(f"Error generating correlation heatmap: {e}")
            else:
                st.warning("Cannot generate heatmap as data preprocessing failed.")

# Model Training page
elif app_mode == "Model Training":
    st.markdown("<h2 class='sub-header'>Model Training and Evaluation</h2>", unsafe_allow_html=True)

    if st.session_state.X is None or st.session_state.y is None or st.session_state.training_columns is None:
        st.warning("⚠️ Please load and preprocess data with a valid binary target variable in 'Data Exploration' first!")
    else:
        st.info(f"Data ready for training: {st.session_state.X.shape[0]} samples, {st.session_state.X.shape[1]} features.")
        st.write("Feature Names:", st.session_state.training_columns)

        # Train/Test Split
        if st.session_state.X_train is None: # Only split if not already done
             try:
                 X_train, X_test, y_train, y_test = train_test_split(
                     st.session_state.X, st.session_state.y,
                     test_size=0.25, # Using 25% for test set
                     random_state=42,
                     stratify=st.session_state.y # Stratify based on target variable
                 )
                 st.session_state.X_train = X_train
                 st.session_state.X_test = X_test
                 st.session_state.y_train = y_train
                 st.session_state.y_test = y_test
                 st.success(f"✅ Data split into training ({X_train.shape[0]} samples) and testing ({X_test.shape[0]} samples).")
             except Exception as e:
                 st.error(f"Error during train/test split: {e}")
                 # Prevent proceeding if split fails
                 st.session_state.X_train = None


        # Train models button
        if st.session_state.X_train is not None: # Check if split was successful
            if st.button("🚀 Train Models", key="train_button"):
                with st.spinner("Training models... This may take a few minutes depending on data size."):
                    try:
                        start_total_time = time.time()
                        # Pass the training data to the train_models function
                        models, scaler, training_times, X_train_scaled, y_train_resampled = train_models(
                            st.session_state.X_train, st.session_state.y_train
                        )

                        st.session_state.models = models
                        st.session_state.scaler = scaler # Store the fitted scaler
                        st.session_state.training_times = training_times
                        st.session_state.X_train_scaled = X_train_scaled # Store scaled training data (potentially needed for SHAP)

                        # Evaluate models on the original (unseen) test set after scaling it
                        st.write("Scaling the test set...")
                        st.session_state.X_test_scaled = st.session_state.scaler.transform(st.session_state.X_test)

                        st.write("Evaluating models on the test set...")
                        st.session_state.model_results = evaluate_models(
                             st.session_state.models,
                             st.session_state.X_test_scaled,
                             st.session_state.y_test
                        )

                        # Create SHAP explainer *only* for XGBoost after training (if XGBoost trained successfully)
                        if 'XGBoost' in st.session_state.models and st.session_state.models['XGBoost']:
                            try:
                                xgb_model = st.session_state.models['XGBoost']['model']
                                # Use a sample of the scaled training data for the explainer background
                                # Adjust sample size as needed for performance vs. accuracy trade-off
                                background_data_sample = shap.sample(st.session_state.X_train_scaled, 100)
                                st.session_state.explainer = shap.TreeExplainer(xgb_model, background_data_sample)
                                st.success("✅ SHAP explainer created for XGBoost model.")
                            except Exception as e:
                                st.error(f"Could not create SHAP explainer for XGBoost: {str(e)}")
                                st.session_state.explainer = None
                        else:
                             st.session_state.explainer = None


                        total_time = time.time() - start_total_time
                        st.success(f"🎉 Models trained and evaluated successfully in {total_time:.2f} seconds!")
                        st.info("Navigate to 'Model Comparison' or 'Risk Prediction'.")

                    except Exception as e:
                        st.error(f"❌ Error during model training or evaluation: {e}")
                        # Clear potentially inconsistent states
                        st.session_state.models = None
                        st.session_state.scaler = None
                        st.session_state.model_results = None


        # Show individual model results if training is complete
        if st.session_state.models is not None and st.session_state.model_results is not None:
            st.markdown("<h3 class='sub-header'>Individual Model Evaluation</h3>", unsafe_allow_html=True)

            model_names = list(st.session_state.models.keys())

            if not model_names: # Check if the list is empty
                 st.warning("⚠️ No models were trained successfully.")
                 # If the list is empty, the code below this 'if' block will be skipped naturally.
                 # No 'return' is needed here.
            else: # Proceed only if model_names is NOT empty
                selected_model = st.selectbox(
                    "Select model to view detailed results:",
                    model_names,
                    key="model_select_eval"
                )

                model_data = st.session_state.models.get(selected_model)
                model_eval = st.session_state.model_results.get(selected_model)

                if model_data and model_eval and model_eval.get('predictions') is not None:
                    st.markdown(f"#### Results for: {selected_model}")

                    # Display metrics
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Performance Metrics")
                        accuracy = accuracy_score(st.session_state.y_test, model_eval['predictions'])
                        st.metric("Accuracy", f"{accuracy:.4f}")

                        st.text("Classification Report:")
                        try:
                            report = classification_report(st.session_state.y_test, model_eval['predictions'], output_dict=True, zero_division=0)
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df.style.format("{:.3f}")) # Format report
                        except Exception as e:
                             st.error(f"Error generating classification report: {e}")

                    with col2:
                        # Confusion matrix
                        st.subheader("Confusion Matrix")
                        cm_fig = plot_confusion_matrix_func(st.session_state.y_test, model_eval['predictions'], selected_model)
                        st.pyplot(cm_fig)
                        plt.close(cm_fig) # Close the figure

                    # Feature importance
                    st.subheader("Feature Importance")
                    importance_fig = plot_feature_importance(
                        model_data['model'],
                        st.session_state.training_columns, # Use the stored feature names
                        selected_model
                    )
                    if importance_fig:
                        st.pyplot(importance_fig)
                        plt.close(importance_fig) # Close the figure

                    # Clinical context analysis
                    st.subheader("Clinical Context Analysis")
                    try:
                        feature_df, clinical_fig1, clinical_fig2, actionability_score = clinical_feature_analysis(
                            model_data['model'],
                            st.session_state.training_columns # Use stored feature names
                        )

                        col_clin1, col_clin2 = st.columns([2, 1])

                        with col_clin1:
                            st.pyplot(clinical_fig1)
                            plt.close(clinical_fig1)

                        with col_clin2:
                            st.pyplot(clinical_fig2)
                            plt.close(clinical_fig2)
                            st.metric("Actionability Score", f"{actionability_score:.3f}",
                                      help="Indicates the relative importance of modifiable/semi-modifiable factors (higher is more actionable). Ranges from 0 to 1.")

                        st.subheader("Top 5 Modifiable Risk Factors")
                        top_modifiable = feature_df[feature_df['Category'] == 'Modifiable'].head(5)
                        if not top_modifiable.empty:
                            st.dataframe(top_modifiable[['Feature', 'Importance']].style.format({'Importance': '{:.4f}'}))
                        else:
                            st.write("No modifiable factors identified in the top features for this model.")

                        st.markdown("""
                        <div class='info-box'>
                        <p><strong>Interpretation:</strong> The 'Actionability Score' reflects the model's reliance on factors that can potentially be changed (lifestyle, medication adherence). Focusing on the 'Top Modifiable Risk Factors' identified by the model may offer the best routes for risk reduction.</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Download feature importance data
                        st.subheader("Download Clinical Analysis Data")
                        csv_clinical = feature_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Feature Importance & Modifiability (CSV)",
                            data=csv_clinical,
                            file_name=f"{selected_model}_clinical_feature_analysis.csv",
                            mime='text/csv',
                        )
                    except Exception as e:
                        st.error(f"Error during clinical context analysis: {e}")

                else:
                    st.warning(f"Results for model '{selected_model}' are not available or evaluation failed.")


# Model Comparison page
elif app_mode == "Model Comparison":
    st.markdown("<h2 class='sub-header'>Model Comparison</h2>", unsafe_allow_html=True)

    if st.session_state.models is None or st.session_state.model_results is None:
        st.warning("⚠️ Please train and evaluate models on the 'Model Training' page first!")
    else:
        st.info("Comparing the performance of all successfully trained models on the test set.")

        # Metrics comparison
        st.subheader("Performance Metrics Comparison")

        metrics = []
        for model_name, eval_data in st.session_state.model_results.items():
            if eval_data and eval_data.get('predictions') is not None and eval_data.get('probabilities') is not None:
                y_true = st.session_state.y_test
                y_pred = eval_data['predictions']
                y_prob = eval_data['probabilities']

                # Calculate metrics safely
                try:
                    acc = accuracy_score(y_true, y_pred)
                    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                    precision_1 = report.get('1', {}).get('precision', 0)
                    recall_1 = report.get('1', {}).get('recall', 0) # Sensitivity for class 1
                    f1_1 = report.get('1', {}).get('f1-score', 0)

                    # Specificity for class 0
                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                    specificity_0 = tn / (tn + fp) if (tn + fp) > 0 else 0

                    fpr, tpr, _ = roc_curve(y_true, y_prob)
                    roc_auc = auc(fpr, tpr)

                    metrics.append({
                        'Model': model_name,
                        'Accuracy': acc,
                        'Sensitivity (Recall_1)': recall_1,
                        'Specificity (Recall_0)': specificity_0, # Approximated by recall of class 0
                        'Precision_1': precision_1,
                        'F1_Score_1': f1_1,
                        'AUC': roc_auc
                    })
                except Exception as e:
                    st.warning(f"Could not calculate metrics for {model_name}: {e}")

        if metrics:
            metrics_df = pd.DataFrame(metrics).set_index('Model')
            st.dataframe(metrics_df.style.format("{:.4f}").highlight_max(axis=0, color='#AED6F1')) # Highlight best in each column

            # Model selection recommendation based on AUC
            st.subheader("Model Recommendation")
            try:
                 best_model_name = metrics_df['AUC'].idxmax()
                 best_auc = metrics_df.loc[best_model_name, 'AUC']
                 st.session_state.best_model_name = best_model_name # Store the name
                 st.markdown(f"""
                 <div class='info-box'>
                 <p>Based on the Area Under the ROC Curve (AUC), the recommended model is:</p>
                 <h3 style='color: #1E88E5;'>{best_model_name}</h3>
                 <p>This model achieves the highest AUC score of <b>{best_auc:.4f}</b> on the test set, indicating strong overall discrimination ability between low and high risk classes.</p>
                 <p><i>Note: The 'best' model might depend on specific needs (e.g., prioritizing sensitivity or specificity). Review all metrics carefully.</i></p>
                 </div>
                 """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Could not determine best model: {e}")
                st.session_state.best_model_name = None

        else:
            st.warning("No model metrics available to compare.")

        # Plotting Comparisons
        st.subheader("Graphical Comparison")
        col_graph1, col_graph2 = st.columns(2)

        with col_graph1:
             st.markdown("#### ROC Curve Comparison")
             roc_fig = plot_roc_curve_func(st.session_state.y_test, st.session_state.model_results)
             st.pyplot(roc_fig)
             plt.close(roc_fig)

        with col_graph2:
            st.markdown("#### Precision-Recall Curve Comparison")
            pr_fig = plot_precision_recall_curve_func(st.session_state.y_test, st.session_state.model_results)
            st.pyplot(pr_fig)
            plt.close(pr_fig)

        # Training time comparison if available
        if st.session_state.training_times:
            st.subheader("Training Time Comparison")
            try:
                times_df = pd.DataFrame.from_dict(st.session_state.training_times, orient='index', columns=['Time (s)'])
                times_df = times_df.reset_index().rename(columns={'index': 'Model'}).sort_values('Time (s)')

                fig_time, ax_time = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Time (s)', y='Model', data=times_df, palette='viridis', ax=ax_time)
                plt.title('Model Training Time Comparison', fontsize=16)
                plt.xlabel('Time (seconds)', fontsize=12)
                plt.ylabel('Model', fontsize=12)
                plt.tight_layout()
                st.pyplot(fig_time)
                plt.close(fig_time)
            except Exception as e:
                st.error(f"Could not plot training times: {e}")


        # SHAP Summary Plot for best model (if it's XGBoost and explainer exists)
        st.subheader("Overall Feature Impact (SHAP Summary)")
        best_model_name_for_shap = st.session_state.get('best_model_name')

        # Allow user to select model for SHAP or default to best
        shap_model_select = st.selectbox(
            "Select model for SHAP Summary Plot (if available):",
            list(st.session_state.models.keys()),
            index=list(st.session_state.models.keys()).index(best_model_name_for_shap) if best_model_name_for_shap in st.session_state.models else 0,
            key="shap_summary_model_select"
        )

        # Currently only works well if the selected model is XGBoost AND the explainer was created during training
        if shap_model_select == 'XGBoost' and st.session_state.explainer is not None:
            if st.button(f"Generate SHAP Summary Plot for {shap_model_select}", key="shap_summary_button"):
                 with st.spinner("Generating SHAP summary plot..."):
                    try:
                        # Use a sample of the scaled test data for the summary plot
                        sample_size = min(500, len(st.session_state.X_test_scaled)) # Use up to 500 samples
                        X_sample_shap = shap.sample(st.session_state.X_test_scaled, sample_size, random_state=42) # Use consistent sample

                        # Calculate SHAP values for the sample
                        shap_values_summary = st.session_state.explainer.shap_values(X_sample_shap)

                        # Create and display SHAP summary plot
                        fig_shap, ax_shap = plt.subplots(figsize=(12, 8)) # Adjust size
                        shap.summary_plot(shap_values_summary, X_sample_shap,
                                          feature_names=st.session_state.training_columns, # Use stored feature names
                                          show=False, plot_size=None) # Let matplotlib handle size via subplots
                        plt.title(f"SHAP Feature Impact Summary for {shap_model_select}", fontsize=16)
                        plt.tight_layout()
                        st.pyplot(fig_shap)
                        plt.close(fig_shap) # Close the figure

                        st.markdown("""
                        <div class='info-box'>
                        <p><strong>How to interpret SHAP Summary Plot:</strong> Each point is a SHAP value for a feature and an instance. Features are ranked by the sum of absolute SHAP values across all samples (global importance).</p>
                        <ul>
                            <li><b>Position on y-axis:</b> Determines the feature.</li>
                            <li><b>Position on x-axis:</b> Shows the impact on the model output (prediction). Positive values push the prediction higher (towards risk=1), negative values push it lower.</li>
                            <li><b>Color:</b> Represents the value of the feature for that instance (Red=High, Blue=Low).</li>
                        </ul>
                        <p>For example, high values (red points) for a feature on the right side indicate that high values of that feature increase the predicted risk.</p>
                        </div>
                        """, unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"❌ Error generating SHAP summary plot for {shap_model_select}: {e}")
                        st.error("Ensure the model and test data are compatible with the SHAP explainer.")

        elif shap_model_select != 'XGBoost':
             st.info(f"Automatic SHAP summary plot generation is currently optimized for the XGBoost model trained by this app. For {shap_model_select}, manual SHAP analysis might be needed.")
        else: # XGBoost selected but explainer is None
             st.warning("XGBoost explainer was not created successfully during training. Cannot generate SHAP summary plot.")


# Risk Prediction page
elif app_mode == "Risk Prediction":
    st.markdown("<h2 class='sub-header'>Personalized Heart Attack Risk Prediction</h2>", unsafe_allow_html=True)

    if st.session_state.models is None or st.session_state.scaler is None or st.session_state.training_columns is None:
        st.warning("⚠️ Please train models on the 'Model Training' page first!")
    else:
        st.write("Use the form below to enter patient information or upload a CSV for batch prediction.")

        # Select model for prediction - default to best if available
        model_list = list(st.session_state.models.keys())
        best_model_idx = 0 # Default to first model
        if st.session_state.best_model_name and st.session_state.best_model_name in model_list:
            best_model_idx = model_list.index(st.session_state.best_model_name)

        model_name = st.selectbox(
            "Select model for prediction:",
            model_list,
            index=best_model_idx,
            key="predict_model_select"
        )

        selected_model = st.session_state.models[model_name]['model']
        # Get the SHAP explainer IF the selected model is XGBoost AND the explainer exists
        explainer_for_prediction = st.session_state.explainer if model_name == "XGBoost" and st.session_state.explainer else None
        if model_name == "XGBoost" and st.session_state.explainer is None:
            st.warning("XGBoost selected, but SHAP explainer is not available for detailed impact analysis.")


        # Create tabs for different input methods
        tab1, tab2 = st.tabs(["👤 Single Patient Input", "📁 Batch Prediction (CSV)"])

        with tab1:
            st.subheader("Enter Patient Information")

            # Use form for better organization and submission handling
            with st.form(key="prediction_form"):
                col1, col2, col3 = st.columns(3)

                # Create dictionary to hold input values dynamically based on original features
                input_data = {}
                # Assuming st.session_state.data holds the original dataframe structure
                original_df_columns = load_sample_data().columns # Use sample data structure as template
                # Find the target column used during training
                target_col_form = None
                possible_targets = ['Heart_Attack_Risk', 'HeartAttackRisk', 'Risk', 'target', 'output']
                for col in possible_targets:
                    if col in original_df_columns:
                         target_col_form = col
                         break

                current_col_index = 0
                cols_per_column = (len(original_df_columns) - (1 if target_col_form else 0)) // 3 + 1
                column_widgets = [col1, col2, col3]


                # Dynamically create input fields based on expected features (from sample or uploaded data)
                if st.session_state.data is not None:
                    df_template = st.session_state.data # Use loaded data structure
                else:
                    df_template = load_sample_data() # Fallback to sample

                original_features = [col for col in df_template.columns if col != target_col_form]

                feature_inputs = {}
                for i, feature in enumerate(original_features):
                     # Determine which column to place the widget in
                     target_column = column_widgets[i // cols_per_column]

                     with target_column:
                        # Infer input type based on dtype in the original dataframe
                        if pd.api.types.is_numeric_dtype(df_template[feature]):
                            min_val = float(df_template[feature].min())
                            max_val = float(df_template[feature].max())
                            mean_val = float(df_template[feature].mean())
                            # Add some buffer to min/max, ensure max > min
                            min_val_input = min_val - (abs(min_val) * 0.1) if min_val > 0 else min_val * 1.1
                            max_val_input = max_val * 1.1 if max_val > 0 else max_val * 0.9
                            if max_val_input <= min_val_input : max_val_input = min_val_input + 1 # Basic check

                            step = 1.0 if pd.api.types.is_integer_dtype(df_template[feature]) else 0.1
                            value_format = "%.0f" if step == 1.0 else "%.1f"

                            # Use number_input or slider based on range/type
                            if max_val - min_val > 20 and step==1.0: # Use number input for large integer ranges
                                 feature_inputs[feature] = st.number_input(
                                     f"{feature}",
                                     min_value=round(min_val_input),
                                     max_value=round(max_val_input),
                                     value=round(mean_val),
                                     step=round(step),
                                     key=f"input_{feature}"
                                 )
                            elif max_val - min_val <= 20 and step >= 1: # Use slider for smaller integer ranges (like Stress Level)
                                feature_inputs[feature] = st.slider(
                                    f"{feature}",
                                     min_value=int(min_val),
                                     max_value=int(max_val),
                                     value=int(round(mean_val)),
                                     step=int(step),
                                     key=f"input_{feature}"
                                )
                            else: # Use number_input for float or slider for small float ranges
                                if max_val - min_val < 50: # Heuristic for using slider for floats
                                     feature_inputs[feature] = st.slider(
                                         f"{feature}",
                                         min_value=round(min_val_input, 1),
                                         max_value=round(max_val_input, 1),
                                         value=round(mean_val, 1),
                                         step=round(step, 1),
                                         key=f"input_{feature}",
                                         format=value_format
                                     )
                                else:
                                     feature_inputs[feature] = st.number_input(
                                         f"{feature}",
                                         min_value=round(min_val_input, 1),
                                         # max_value=round(max_val_input, 1), # Max value causes issues sometimes
                                         value=round(mean_val, 1),
                                         step=round(step, 1),
                                         key=f"input_{feature}",
                                         format=value_format
                                     )

                        elif pd.api.types.is_object_dtype(df_template[feature]) or pd.api.types.is_categorical_dtype(df_template[feature]):
                            options = df_template[feature].unique().tolist()
                            # Try to find a reasonable default, like 'No' or 'Never' or the mode
                            default_option = options[0]
                            if 'No' in options: default_option = 'No'
                            if 'Never' in options: default_option = 'Never'
                            if 'Female' in options and 'Male' in options: default_option = 'Female' # Example default

                            feature_inputs[feature] = st.selectbox(f"{feature}", options=options, index=options.index(default_option), key=f"input_{feature}")
                        else:
                             # Fallback for other types
                             feature_inputs[feature] = st.text_input(f"{feature}", value=str(df_template[feature].iloc[0]), key=f"input_{feature}")


                submitted = st.form_submit_button("⚡ Predict Heart Attack Risk")

            # --- Prediction Execution (outside the form) ---
            if submitted:
                with st.spinner("Calculating risk..."):
                    try:
                        # Make prediction using selected model and the input data
                        prediction, probability, shap_explanation = predict_individual_risk(
                            feature_inputs, # Pass the dictionary from the form
                            selected_model,
                            st.session_state.training_columns, # Crucial: use columns from training
                            st.session_state.scaler,
                            explainer_for_prediction,
                             st.session_state.training_columns # Pass original feature names (post-dummies)
                        )

                        if prediction is not None:
                            st.markdown("<h3 class='sub-header'>Risk Assessment Results</h3>", unsafe_allow_html=True)

                            col_res1, col_res2 = st.columns([1, 2]) # Adjust column ratio

                            with col_res1:
                                st.markdown("#### Overall Risk")
                                if prediction == 1:
                                    st.markdown("<div class='risk-high'><p class='risk-text'>HIGH RISK</p></div>", unsafe_allow_html=True)
                                else:
                                    st.markdown("<div class='risk-low'><p class='risk-text'>LOW RISK</p></div>", unsafe_allow_html=True)

                                # Probability gauge - simple text version for robustness
                                st.metric("Risk Probability", f"{probability*100:.1f}%")

                                # Risk level description
                                if probability < 0.25:
                                    risk_level = "Low"
                                    recommendation = "Risk appears low. Continue healthy habits."
                                elif probability < 0.5:
                                    risk_level = "Moderate"
                                    recommendation = "Risk is moderate. Consider lifestyle improvements and discuss with your doctor."
                                elif probability < 0.75:
                                    risk_level = "High"
                                    recommendation = "Risk appears high. Consult with a healthcare provider for evaluation and guidance."
                                else:
                                    risk_level = "Very High"
                                    recommendation = "Risk appears very high. Urgent medical consultation is strongly recommended."

                                st.metric("Risk Level Category", risk_level)
                                st.info(f"💡 Recommendation: {recommendation}")

                            with col_res2:
                                # Feature impact (SHAP Waterfall)
                                st.markdown("#### Key Factors Influencing Prediction")

                                if shap_explanation is not None and model_name == "XGBoost":
                                    try:
                                        fig_waterfall, ax_waterfall = plt.subplots(figsize=(10, 6)) # Adjust size
                                        shap.waterfall_plot(shap_explanation, max_display=10, show=False)
                                        plt.title(f"SHAP Waterfall Plot for {model_name} Prediction", fontsize=14)
                                        plt.tight_layout()
                                        st.pyplot(fig_waterfall)
                                        plt.close(fig_waterfall) # Close figure

                                        st.markdown("""
                                        <div class='info-box' style='font-size: 0.9em;'>
                                        <p><strong>How to interpret Waterfall Plot:</strong> This plot shows how each feature contributes to moving the prediction from the baseline (average prediction) to the final prediction for this individual.</p>
                                        <ul><li><b>Red bars:</b> Features pushing the risk prediction higher.</li><li><b>Blue bars:</b> Features pushing the risk prediction lower.</li><li><b>Length of bar:</b> Magnitude of the feature's impact.</li></ul>
                                        <p>Focus on the most impactful factors (longest bars), especially the red ones if the risk is high, to understand potential areas for intervention.</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    except Exception as e:
                                        st.error(f"Error plotting SHAP waterfall: {e}")
                                        st.info("Detailed feature impact analysis using SHAP waterfall plot is not available for this prediction.")

                                elif model_name == "XGBoost":
                                     st.warning("SHAP explainer not available for XGBoost. Cannot show waterfall plot.")
                                else:
                                     st.info(f"Detailed SHAP feature impact analysis is currently only configured for XGBoost. Showing general feature importance for {model_name} instead.")
                                     # Show general feature importance as fallback
                                     importance_fig_pred = plot_feature_importance(
                                         selected_model,
                                         st.session_state.training_columns,
                                         f"General Importance - {model_name}",
                                         top_n=10 # Show top 10
                                     )
                                     if importance_fig_pred:
                                        st.pyplot(importance_fig_pred)
                                        plt.close(importance_fig_pred)


                            # Personalized recommendations (simplified example)
                            st.markdown("<h3 class='sub-header'>Personalized Considerations</h3>", unsafe_allow_html=True)
                            st.markdown("""
                                <div class='info-box'>
                                <ul>
                                    <li>Review the factors identified as increasing risk (red bars in SHAP plot if available, or top factors from general importance).</li>
                                    <li>Discuss modifiable factors like <b>Blood Pressure, Cholesterol, BMI, Smoking status, Exercise levels, and Stress</b> with your healthcare provider.</li>
                                    <li>Non-modifiable factors like <b>Age, Sex, and Family History</b> are important for context but cannot be changed.</li>
                                </ul>
                                </div>
                            """, unsafe_allow_html=True)


                            # Add disclaimer
                            st.markdown("""
                            <div class='info-box' style='margin-top: 20px; border-color: #FFA726; background-color: #FFF3E0;'>
                            <p><strong>⚠️ Disclaimer:</strong> This prediction is based on a machine learning model and the data provided. It is for informational purposes only and <strong>does not constitute medical advice</strong>. It cannot replace a thorough evaluation by a qualified healthcare professional. Always consult with your doctor or other qualified health provider regarding any questions you may have about a medical condition or treatment options.</p>
                            </div>
                            """, unsafe_allow_html=True)

                        else:
                            st.error("❌ Prediction failed. Please check input values and model status.")

                    except Exception as e:
                        st.error(f"❌ An error occurred during prediction: {e}")
                        st.error("Please ensure all input fields are filled correctly and the model was trained successfully.")

        # --- Batch Prediction Tab ---
        with tab2:
            st.subheader("Upload Patient Data CSV for Batch Prediction")
            st.write(f"Upload a CSV file with patient records. Ensure columns match the features used for training (or the sample data structure): `{', '.join(original_features)}`")

            uploaded_csv = st.file_uploader("Upload CSV for Batch Prediction", type=["csv"], key="batch_uploader")

            if uploaded_csv is not None:
                try:
                    patients_df_original = pd.read_csv(uploaded_csv)
                    st.write("Preview of uploaded data:")
                    st.dataframe(patients_df_original.head())

                    if st.button("🚀 Generate Batch Predictions", key="batch_predict_button"):
                        with st.spinner("Processing batch predictions..."):
                            patients_df = patients_df_original.copy() # Work on a copy

                             # --- Preprocessing Logic for Batch ---
                            # 1. Handle missing values (using median/mode from training data if available, or from batch itself)
                            for col in patients_df.columns:
                                if patients_df[col].isnull().any():
                                    if pd.api.types.is_numeric_dtype(patients_df[col]):
                                        # Ideally, use median from training data (st.session_state.X[col].median())
                                        # Fallback to batch median if training data stats not stored
                                        batch_median = patients_df[col].median()
                                        patients_df[col] = patients_df[col].fillna(batch_median)
                                    elif pd.api.types.is_object_dtype(patients_df[col]):
                                        batch_mode = patients_df[col].mode()
                                        patients_df[col] = patients_df[col].fillna(batch_mode[0] if not batch_mode.empty else 'Unknown')

                            # 2. Apply get_dummies
                            patients_df_dummies = pd.get_dummies(patients_df, drop_first=True, dummy_na=False)

                            # 3. Align columns with training data
                            patients_df_reindexed = patients_df_dummies.reindex(columns=st.session_state.training_columns, fill_value=0)

                            # 4. Scale using the fitted scaler
                            try:
                                patients_scaled = st.session_state.scaler.transform(patients_df_reindexed)
                            except ValueError as ve:
                                st.error(f"Scaling error: {ve}")
                                st.error("This often means the columns in your uploaded CSV do not match the columns the model was trained on after preprocessing.")
                                st.error(f"Model expects columns: {st.session_state.training_columns}")
                                st.error(f"Your CSV resulted in columns (after dummies/reindex): {patients_df_reindexed.columns.tolist()}")
                                st.stop() # Stop execution here
                            except Exception as e:
                                st.error(f"An unexpected error occurred during scaling: {e}")
                                st.stop()


                            # --- Make Predictions ---
                            predictions = selected_model.predict(patients_scaled)
                            probabilities = selected_model.predict_proba(patients_scaled)[:, 1]

                            # Add predictions back to the original DataFrame copy
                            patients_df['Predicted_Risk_Label'] = predictions
                            patients_df['Predicted_Risk_Probability'] = probabilities
                            patients_df['Predicted_Risk_Category'] = pd.cut(
                                patients_df['Predicted_Risk_Probability'],
                                bins=[0, 0.25, 0.5, 0.75, 1.0],
                                labels=['Low', 'Moderate', 'High', 'Very High'],
                                include_lowest=True
                             )

                            # Display results
                            st.subheader("Batch Prediction Results")
                            st.dataframe(patients_df)

                            # Download results
                            csv_results = patients_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Predictions CSV",
                                data=csv_results,
                                file_name=f"{model_name}_batch_predictions.csv",
                                mime='text/csv',
                             )

                            # Summary statistics
                            st.subheader("Batch Summary")
                            high_risk_count = sum(predictions == 1)
                            total_patients = len(predictions)
                            high_risk_percentage = (high_risk_count / total_patients) * 100 if total_patients > 0 else 0

                            col_sum1, col_sum2 = st.columns(2)
                            with col_sum1:
                                st.metric("Total Patients Processed", total_patients)
                                st.metric("Patients Predicted High Risk", high_risk_count)
                            with col_sum2:
                                st.metric("High Risk Percentage", f"{high_risk_percentage:.1f}%")
                                avg_prob = np.mean(probabilities) * 100 if total_patients > 0 else 0
                                st.metric("Average Risk Probability", f"{avg_prob:.1f}%")

                            # Distribution of risk probabilities
                            st.subheader("Distribution of Predicted Risk Probabilities")
                            fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
                            sns.histplot(probabilities, bins=20, kde=True, ax=ax_dist, color='skyblue')
                            plt.title("Distribution of Risk Probabilities in Batch", fontsize=14)
                            plt.xlabel("Predicted Risk Probability", fontsize=12)
                            plt.ylabel("Number of Patients", fontsize=12)
                            plt.tight_layout()
                            st.pyplot(fig_dist)
                            plt.close(fig_dist)


                except Exception as e:
                    st.error(f"❌ Error processing batch CSV file: {e}")
                    st.error("Please ensure your CSV has the required features and is formatted correctly.")


# Footer
st.markdown("""---""") # Add a visual separator
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 15px; background-color: #F0F2F6; border-radius: 5px; font-size: 0.9em;">
<p>Created using Streamlit, Scikit-learn, XGBoost, SHAP, and ❤️</p>
<p>This application is intended for educational and informational purposes only.
<strong>It is not a substitute for professional medical advice, diagnosis, or treatment.</strong>
Always consult with a qualified healthcare provider for any health concerns or before making any decisions related to your health or treatment.</p>
</div>
""", unsafe_allow_html=True)

# Optional: Clear Matplotlib cache sometimes helps with memory
# plt.close('all')