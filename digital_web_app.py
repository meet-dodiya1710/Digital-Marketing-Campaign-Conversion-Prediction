import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Streamlit App
st.title("Digital Marketing Campaign Classification")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Data Info
    st.subheader("Dataset Information")
    st.write(data.info())
    
    # Data Overview
    st.subheader("Dataset Description")
    st.write(data.describe())

    # Value Counts for the target variable
    st.subheader("Target Variable (Conversion) Distribution")
    st.write(data['Conversion'].value_counts())
    
    # Univariate Analysis for Continuous Features
    st.subheader("Distribution of Continuous Features")
    continuous_features = data.select_dtypes(include=['int64', 'float64']).columns
    continuous_features = continuous_features[continuous_features != 'Conversion']
    
    fig, axes = plt.subplots(nrows=int(np.ceil(len(continuous_features) / 3)), ncols=3, figsize=(18, 4 * len(continuous_features)//3))
    axes = axes.flatten()
    
    for i, column in enumerate(continuous_features):
        sns.histplot(data[column], kde=True, color='green', bins=30, ax=axes[i])
        axes[i].set_title(f'Distribution of {column}')
    
    st.pyplot(fig)

    # Univariate Analysis for Categorical Features
    st.subheader("Distribution of Categorical Features")
    categorical_features = data.select_dtypes(include=['object']).columns
    
    fig, axes = plt.subplots(nrows=int(np.ceil(len(categorical_features) / 2)), ncols=2, figsize=(14, 6))
    axes = axes.flatten()
    
    for i, column in enumerate(categorical_features):
        sns.countplot(x=data[column], ax=axes[i])
        axes[i].set_title(f'Distribution of {column}')
    
    st.pyplot(fig)

    # Feature Engineering and Preprocessing
    data.drop(['CustomerID', 'ConversionRate'], axis=1, inplace=True)
    st.subheader("Missing Data")
    st.write(data.isnull().sum())

    data['EngagementScore'] = data['PagesPerVisit'] * data['TimeOnSite']
    data['AdEffectiveness'] = data['AdSpend'] * data['ClickThroughRate']
    data['CostPerConversion'] = data['AdSpend'] / (data['Conversion'] + 1)
    data['AvgPurchaseValue'] = data['Income'] / (data['PreviousPurchases'] + 1)
    data['LoyaltyScore'] = data['LoyaltyPoints'] / (data['PreviousPurchases'] + 1)
    data['LogAdSpend'] = np.log1p(data['AdSpend'])
    data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 25, 45, 65, 100], labels=['Youth', 'Adult', 'Middle-aged', 'Senior'])
    data['WebsiteVisits'] = data['WebsiteVisits'].replace(0, data['WebsiteVisits'].median())
    data['PagesPerVisit'] = data['PagesPerVisit'].replace(0, data['PagesPerVisit'].median())
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

    numeric_features = ['Age', 'Income', 'AdSpend', 'ClickThroughRate', 'WebsiteVisits', 'PagesPerVisit', 'TimeOnSite', 'SocialShares', 'EmailOpens', 'LogAdSpend']
    from sklearn.preprocessing import LabelEncoder
    categorical_cols = data.select_dtypes(include=['object']).columns
    label_encoder = LabelEncoder()
    for col in categorical_cols:
        data[col] = label_encoder.fit_transform(data[col])
    scaler = StandardScaler()
    data[numeric_features] = scaler.fit_transform(data[numeric_features])

    X = data.drop(['Conversion', 'AgeGroup'], axis=1)
    y = data['Conversion']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    model_choice = st.selectbox("Choose a Model", ["Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost", "Logistic Regression"])

    def train_model(model, param_grid):
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train_resampled, y_train_resampled)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        return accuracy, report, conf_matrix

    accuracy, report, conf_matrix = None, None, None

    if model_choice == "Decision Tree":
        dtree_params = {'max_depth': [1, 2, 3, 5], 'min_samples_split': [10, 20, 50], 'min_samples_leaf': [5, 10, 20], 'criterion': ['gini', 'entropy']}
        accuracy, report, conf_matrix = train_model(DecisionTreeClassifier(random_state=42), dtree_params)

    elif model_choice == "Random Forest":
        rf_params = {'n_estimators': [100, 200], 'max_depth': [10, 20], 'min_samples_split': [2, 5]}
        accuracy, report, conf_matrix = train_model(RandomForestClassifier(random_state=42), rf_params)

    elif model_choice == "Gradient Boosting":
        gb_params = {'n_estimators': [7, 12], 'learning_rate': [0.3, 0.4], 'max_depth': [1, 2]}
        accuracy, report, conf_matrix = train_model(GradientBoostingClassifier(random_state=42), gb_params)

    elif model_choice == "XGBoost":
        xgb_params = {'n_estimators': [5, 10, 20], 'learning_rate': [0.1, 0.2], 'max_depth': [3, 4, 5]}
        accuracy, report, conf_matrix = train_model(XGBClassifier(random_state=42), xgb_params)

    elif model_choice == "Logistic Regression":
        logreg_params = {'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs', 'liblinear'], 'max_iter': [100, 200, 500]}
        accuracy, report, conf_matrix = train_model(LogisticRegression(random_state=42), logreg_params)

    results = {
        "Model": [model_choice],
        "Accuracy": [accuracy]
    }

    results_df = pd.DataFrame(results)
    st.subheader("Model Comparison Results")
    st.table(results_df)

    st.subheader(f"Classification Report ({model_choice})")
    report_df = pd.DataFrame(report).transpose()
    st.table(report_df)

    st.subheader(f"Confusion Matrix ({model_choice})")
    st.write(conf_matrix)