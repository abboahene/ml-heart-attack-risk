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
import shap
import base64
from io import BytesIO
import time
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

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
    .model-metrics {
        display: flex;
        justify-content: space-between;
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
        df = pd.read_csv("sample_data.csv")
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

    with st.spinner('Training Logistic Regression model...'):
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train_scaled, y_train)
        models['Logistic Regression'] = {
            'model': lr,
            'predictions': lr.predict(X_test_scaled),
            'probabilities': lr.predict_proba(X_test_scaled)[:, 1]
        }

    with st.spinner('Training Support Vector Machine model...'):
        svm = SVC(random_state=42, probability=True)
        svm.fit(X_train_scaled, y_train)
        models['SVM'] = {
            'model': svm,
            'predictions': svm.predict(X_test_scaled),
            'probabilities': svm.predict_proba(X_test_scaled)[:, 1]
        }

    with st.spinner('Training Gradient Boosting model...'):
        gb = GradientBoostingClassifier(random_state=42)
        gb.fit(X_train_scaled, y_train)
        models['Gradient Boosting'] = {
            'model': gb,
            'predictions': gb.predict(X_test_scaled),
            'probabilities': gb.predict_proba(X_test_scaled)[:, 1]
        }

    with st.spinner('Training Neural Network model...'):
        nn = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        nn.fit(X_train_scaled, y_train)
        models['Neural Network'] = {
            'model': nn,
            'predictions': nn.predict(X_test_scaled),
            'probabilities': nn.predict_proba(X_test_scaled)[:, 1]
        }

    # Stacked Classifier
    with st.spinner('Training Stacked Classifier...'):
        base_learners = [
            ('lr', lr),
            ('rf', rf),
            ('svm', svm)
        ]
        stack = StackingClassifier(
            estimators=base_learners,
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            cv=5,
            n_jobs=-1
        )
        stack.fit(X_train_scaled, y_train)
        models['Stacked Classifier'] = {
            'model': stack,
            'predictions': stack.predict(X_test_scaled),
            'probabilities': stack.predict_proba(X_test_scaled)[:, 1]
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

def plot_roc_curve(y_true, probas, models):
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
    # Check if model has feature_importances_ attribute
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
        # For linear models like Logistic Regression
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
        return None  # Model doesn't support direct feature importance

def clinical_feature_analysis(model, feature_names):
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
    ["Home", "Data Exploration", "Model Training", "Model Comparison", "Risk Prediction"]
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
        - Training and comparison of multiple machine learning models:
          - XGBoost
          - Random Forest
          - Logistic Regression
          - Support Vector Machine (SVM) 
          - Gradient Boosting
          - Neural Network (MLP)
        - Clinical context analysis of risk factors
        - Personalized risk prediction
        
        ### How to use:
        1. Start by uploading your dataset or use our sample data
        2. Explore the data visualizations
        3. Train and evaluate models
        4. Compare model performance
        5. Get personalized risk predictions
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
                    start_time = time.time()
                    models, X_train, X_test, y_train, y_test, scaler, X_train_scaled, X_test_scaled = train_models(
                        st.session_state.X, st.session_state.y
                    )
                    training_time = time.time() - start_time
                    
                    # Store in session state
                    st.session_state.models = models
                    st.session_state.scaler = scaler
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    st.session_state.X_train_scaled = X_train_scaled
                    st.session_state.X_test_scaled = X_test_scaled
                    
                    # Create SHAP explainer for XGBoost
                    st.session_state.explainer = shap.Explainer(models['XGBoost']['model'], X_train_scaled)
                    
                    st.success(f"Models trained successfully in {training_time:.2f} seconds!")
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

# Model Comparison page
elif app_mode == "Model Comparison":
    st.markdown("<h2 class='sub-header'>Model Comparison</h2>", unsafe_allow_html=True)
    
    if st.session_state.models is None:
        st.warning("Please train models first!")
    else:
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
                'AUC': roc_auc
            })
        
        metrics_df = pd.DataFrame(metrics)
        metrics_df = metrics_df.set_index('Model')
        
        # Format the values
        formatted_df = metrics_df.applymap(lambda x: f"{x:.4f}")
        
        # Add styling
        st.dataframe(formatted_df.style.highlight_max(axis=0, color='#AED6F1'))
        
        # ROC Curve Comparison
        st.subheader("ROC Curve Comparison")
        
        # Prepare probabilities dict for plotting
        probas = {model_name: model_data['probabilities'] for model_name, model_data in st.session_state.models.items()}
        
        # Plot ROC curves
        roc_fig = plot_roc_curve(st.session_state.y_test, probas, st.session_state.models)
        st.pyplot(roc_fig)
        
        # Precision-Recall Curve
        st.subheader("Precision-Recall Curve Comparison")
        pr_fig = plot_precision_recall_curve(st.session_state.y_test, probas, st.session_state.models)
        st.pyplot(pr_fig)
        
        # Training time comparison if available
        if 'training_times' in st.session_state:
            st.subheader("Training Time Comparison")
            times_df = pd.DataFrame(st.session_state.training_times)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(times_df['Model'], times_df)
            sns.barplot(x='Time (s)', y='Model', data=times_df, palette='viridis', ax=ax)
            plt.title('Model Training Time Comparison')
            plt.xlabel('Time (seconds)')
            plt.tight_layout()
            st.pyplot(fig)
        
        # Model selection recommendation
        st.subheader("Model Recommendation")
        
        # Find the best model based on AUC (can change to other metrics)
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
        st.subheader(f"SHAP Feature Impact for {best_model_name}")
        
        if st.button("Generate SHAP Analysis (may take a minute)"):
            with st.spinner("Generating SHAP values..."):
                try:
                    # Create explainer for the best model
                    best_model = st.session_state.models[best_model_name]['model']
                    
                    # Use a sample of the test data for computation efficiency
                    sample_size = min(100, len(st.session_state.X_test_scaled))
                    X_sample = st.session_state.X_test_scaled[:sample_size]
                    
                    if best_model_name == 'XGBoost':
                        explainer = st.session_state.explainer
                    else:
                        explainer = shap.Explainer(best_model, st.session_state.X_train_scaled)
                    
                    shap_values = explainer(X_sample)
                    
                    # Create and display SHAP summary plot
                    fig, ax = plt.subplots(figsize=(10, 8))
                    shap.summary_plot(shap_values, X_sample, feature_names=st.session_state.feature_names, show=False)
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

# Risk Prediction page
elif app_mode == "Risk Prediction":
    st.markdown("<h2 class='sub-header'>Personalized Heart Attack Risk Prediction</h2>", unsafe_allow_html=True)
    
    if st.session_state.models is None or st.session_state.scaler is None:
        st.warning("Please train models first!")
    else:
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
                input_data['Race'] = st.selectbox("Race", ["White", "Black", "Asian", "Hispanic", "Other"])
                
            # Health metrics
            with col2:
                st.markdown("##### Health Metrics")
                input_data['Cholesterol'] = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=500, value=200)
                input_data['Blood Pressure'] = st.number_input("Systolic Blood Pressure (mmHg)", min_value=80, max_value=220, value=120)
                input_data['Heart Rate'] = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=75)
                input_data['BMI'] = st.number_input("BMI", min_value=10.0, max_value=50.0, value=24.5, step=0.1)
                
            # Lifestyle and conditions
            with col3:
                st.markdown("##### Lifestyle & Conditions")
                input_data['Smoking'] = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
                input_data['Alcohol Consumption'] = st.selectbox("Alcohol Consumption", ["None", "Moderate", "Heavy"])
                input_data['Exercise Hours Per Week'] = st.slider("Exercise (hours/week)", 0.0, 20.0, 3.0, 0.5)
                input_data['Diabetes'] = st.selectbox("Diabetes", ["No", "Yes"])
                input_data['Family History'] = st.selectbox("Family History of Heart Disease", ["No", "Yes"])
                
            # More health details
            st.markdown("##### Additional Details")
            col1, col2 = st.columns(2)
            
            with col1:
                input_data['Sleep Hours'] = st.slider("Sleep (hours/day)", 3.0, 12.0, 7.0, 0.5)
                input_data['Physical Activity Days Per Week'] = st.slider("Active Days Per Week", 0, 7, 3)
                input_data['Medication Use'] = st.multiselect(
                    "Medications", 
                    ["None", "Anti-hypertensive", "Cholesterol-lowering", "Anti-diabetic", "Other"],
                    default=["None"]
                )
                if "None" in input_data['Medication Use'] and len(input_data['Medication Use']) > 1:
                    input_data['Medication Use'].remove("None")
                input_data['Medication Use'] = ", ".join(input_data['Medication Use']) if input_data['Medication Use'] else "None"
                
            with col2:
                input_data['Stress Level'] = st.slider("Stress Level (1-10)", 1, 10, 5)
                input_data['Sedentary Hours Per Day'] = st.slider("Sedentary Hours/Day", 0.0, 24.0, 8.0, 0.5)
                input_data['Previous Heart Problems'] = st.selectbox("Previous Heart Problems", ["No", "Yes"])
                input_data['Diet'] = st.selectbox(
                    "Diet Quality", 
                    ["Poor", "Average", "Good", "Excellent"]
                )
            
            # Predict button
            if st.button("Predict Heart Attack Risk"):
                with st.spinner("Calculating risk..."):
                    try:
                        # Make prediction using selected model
                        prediction, probability, shap_values = predict_individual_risk(
                            input_data, 
                            selected_model,
                            st.session_state.X.columns,
                            st.session_state.scaler,
                            st.session_state.explainer if model_name == "XGBoost" else None
                        )
                        
                        # Display results
                        st.subheader("Risk Assessment Results")
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            if prediction == 1:
                                st.markdown("""
                                <div style="background-color:#FF5252; padding:20px; border-radius:10px; text-align:center;">
                                <h2 style="color:white;">HIGH RISK</h2>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                <div style="background-color:#4CAF50; padding:20px; border-radius:10px; text-align:center;">
                                <h2 style="color:white;">LOW RISK</h2>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Probability gauge
                            fig, ax = plt.subplots(figsize=(4, 0.8))
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
                            
                            st.pyplot(fig)
                            
                            # Risk level description
                            if probability < 0.25:
                                risk_level = "Low Risk"
                                recommendation = "Continue with healthy lifestyle."
                            elif probability < 0.5:
                                risk_level = "Moderate Risk"
                                recommendation = "Consider lifestyle improvements."
                            elif probability < 0.75:
                                risk_level = "High Risk"
                                recommendation = "Consult with healthcare provider soon."
                            else:
                                risk_level = "Very High Risk"
                                recommendation = "Urgent medical consultation recommended!"
                            
                            st.metric("Risk Probability", f"{probability*100:.1f}%")
                            st.metric("Risk Level", risk_level)
                            st.info(f"Recommendation: {recommendation}")
                        
                        with col2:
                            # Feature impact
                            st.subheader("Feature Impact on Prediction")
                            
                            if shap_values is not None:
                                # Create waterfall plot
                                fig, ax = plt.subplots(figsize=(10, 8))
                                shap.waterfall_plot(shap_values[0], max_display=10, show=False)
                                plt.title("Top Factors Influencing Risk Prediction")
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                st.markdown("""
                                <div class='info-box'>
                                <p><strong>How to interpret:</strong> Red bars increase risk, blue bars decrease risk. 
                                The length of each bar shows the magnitude of impact.</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.info("Detailed feature impact analysis not available for this model.")
                        
                        # Personalized recommendations
                        st.subheader("Personalized Recommendations")
                        
                        # Define risk factors and recommendations
                        modifiable_factors = {
                            'Cholesterol': {
                                'high': {'threshold': 200, 'recommendation': "Consider dietary changes to reduce cholesterol and speak with a healthcare provider about medication options if appropriate."},
                                'normal': {'recommendation': "Maintain healthy cholesterol levels through a balanced diet."}
                            },
                            'Blood Pressure': {
                                'high': {'threshold': 130, 'recommendation': "Monitor blood pressure regularly. Consider the DASH diet, reducing sodium, regular exercise, and stress management."},
                                'normal': {'recommendation': "Continue maintaining healthy blood pressure with regular monitoring."}
                            },
                            'BMI': {
                                'high': {'threshold': 25, 'recommendation': "Focus on achieving a healthier weight through balanced nutrition and regular physical activity."},
                                'normal': {'recommendation': "Maintain healthy weight through continued balanced diet and regular exercise."}
                            },
                            'Smoking': {
                                'risk': {'values': ['Current'], 'recommendation': "Quitting smoking is one of the most important steps to reduce heart attack risk. Consider cessation programs or speak with a healthcare provider about support options."},
                                'former': {'values': ['Former'], 'recommendation': "Great job quitting smoking! Your risk continues to decrease the longer you remain smoke-free."},
                                'none': {'recommendation': "Continue to avoid smoking to maintain your lower risk profile."}
                            },
                            'Exercise Hours Per Week': {
                                'low': {'threshold': 2.5, 'comparison': '<', 'recommendation': "Aim for at least 150 minutes (2.5 hours) of moderate exercise per week, as recommended by health guidelines."},
                                'normal': {'recommendation': "Great job staying active! Continue your regular exercise routine."}
                            },
                            'Sleep Hours': {
                                'poor': {'threshold_low': 7, 'threshold_high': 9, 'comparison': 'outside', 'recommendation': "Try to achieve 7-9 hours of quality sleep per night for optimal heart health."},
                                'good': {'recommendation': "Continue maintaining healthy sleep habits."}
                            },
                            'Stress Level': {
                                'high': {'threshold': 7, 'recommendation': "Consider stress management techniques such as mindfulness, meditation, or speaking with a mental health professional."},
                                'normal': {'recommendation': "Continue practicing stress management for heart health."}
                            }
                        }
                        
                        recommendations = []
                        
                        # Generate personalized recommendations based on input data
                        for factor, conditions in modifiable_factors.items():
                            if factor in input_data:
                                value = input_data[factor]
                                
                                # Handle different types of comparisons
                                if 'high' in conditions and 'threshold' in conditions['high']:
                                    threshold = conditions['high']['threshold']
                                    comparison = conditions['high'].get('comparison', '>')
                                    
                                    if comparison == '>' and value > threshold:
                                        recommendations.append(f"**{factor}**: {conditions['high']['recommendation']}")
                                    elif comparison == '<' and value < threshold:
                                        recommendations.append(f"**{factor}**: {conditions['low']['recommendation']}")
                                    elif comparison == 'outside' and (value < conditions['poor']['threshold_low'] or value > conditions['poor']['threshold_high']):
                                        recommendations.append(f"**{factor}**: {conditions['poor']['recommendation']}")
                                    else:
                                        if 'normal' in conditions:
                                            recommendations.append(f"**{factor}**: {conditions['normal']['recommendation']}")
                                
                                # Handle categorical variables
                                elif 'risk' in conditions and 'values' in conditions['risk']:
                                    if value in conditions['risk']['values']:
                                        recommendations.append(f"**{factor}**: {conditions['risk']['recommendation']}")
                                    elif 'former' in conditions and value in conditions['former'].get('values', []):
                                        recommendations.append(f"**{factor}**: {conditions['former']['recommendation']}")
                                    else:
                                        if 'none' in conditions:
                                            recommendations.append(f"**{factor}**: {conditions['none']['recommendation']}")
                        
                        # Display recommendations
                        for rec in recommendations:
                            st.write(rec)
                        
                        # Add disclaimer
                        st.markdown("""
                        <div class='info-box' style='margin-top: 20px;'>
                        <p><strong>Disclaimer:</strong> This prediction is for informational purposes only and should not 
                        replace professional medical advice. Always consult with a healthcare provider for proper diagnosis 
                        and treatment recommendations.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error making prediction: {e}")
                        st.error("Make sure all required features are provided and in the correct format.")
        
        with tab2:
            st.subheader("Upload Patient Data CSV")
            st.write("Upload a CSV file with multiple patient records for batch prediction.")
            
            uploaded_csv = st.file_uploader("Upload CSV", type=["csv"])
            
            if uploaded_csv is not None:
                try:
                    patients_df = pd.read_csv(uploaded_csv)
                    st.write("Preview of uploaded data:")
                    st.write(patients_df.head())
                    
                    if st.button("Generate Batch Predictions"):
                        with st.spinner("Processing batch predictions..."):
                            # Preprocess the data similar to training data
                            patients_df_processed = preprocess_data(patients_df)
                            
                            # Ensure columns match training data
                            for col in st.session_state.X.columns:
                                if col not in patients_df_processed.columns:
                                    patients_df_processed[col] = 0
                            
                            # Select only columns used in training
                            patients_df_final = patients_df_processed[st.session_state.X.columns]
                            
                            # Scale the data
                            patients_scaled = st.session_state.scaler.transform(patients_df_final)
                            
                            # Make predictions
                            predictions = selected_model.predict(patients_scaled)
                            probabilities = selected_model.predict_proba(patients_scaled)[:, 1]
                            
                            # Add predictions to original data
                            patients_df['Risk Prediction'] = predictions
                            patients_df['Risk Probability'] = probabilities
                            
                            # Display results
                            st.subheader("Batch Prediction Results")
                            st.write(patients_df)
                            
                            # Download results
                            csv = patients_df.to_csv(index=False)
                            b64 = base64.b64encode(csv.encode()).decode()
                            href = f'<a href="data:file/csv;base64,{b64}" download="heart_risk_predictions.csv">Download Predictions CSV</a>'
                            st.markdown(href, unsafe_allow_html=True)
                            
                            # Summary statistics
                            high_risk_count = sum(predictions == 1)
                            high_risk_percentage = (high_risk_count / len(predictions)) * 100
                            
                            st.subheader("Batch Summary")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Total Patients", len(predictions))
                                st.metric("High Risk Patients", high_risk_count)
                            with col2:
                                st.metric("High Risk Percentage", f"{high_risk_percentage:.1f}%")
                                avg_prob = np.mean(probabilities) * 100
                                st.metric("Average Risk Probability", f"{avg_prob:.1f}%")
                            
                            # Distribution of risk probabilities
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.histplot(probabilities, bins=20, kde=True, ax=ax)
                            plt.title("Distribution of Risk Probabilities")
                            plt.xlabel("Risk Probability")
                            plt.ylabel("Count")
                            st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error processing CSV file: {e}")
                    st.error("Please ensure your CSV has the required features in the correct format.")

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 20px; background-color: #F0F2F6; border-radius: 5px;">
<p>Created with ❤️ for healthcare professionals and patients</p>
<p>This application is for educational and informational purposes only.</p>
<p>Always consult with healthcare professionals for medical advice.</p>
</div>
""", unsafe_allow_html=True)