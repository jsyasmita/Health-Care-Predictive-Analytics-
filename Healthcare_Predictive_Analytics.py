import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Upload the dataset
st.title("Heart Failure Prediction - Predictive Analytics Dashboard")

# File uploader for the dataset
uploaded_file = st.file_uploader("Upload the Training Dataset (CSV format)", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(data.head())

    # Step 2: Pre-process the data
    # Handle missing values if any
    data = data.dropna()

    # Select relevant features for prediction
    features = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 
                'serum_sodium', 'sex', 'smoking']
    X = data[features]
    y = data['DEATH_EVENT']  # Target variable: 1 if the patient died, 0 otherwise

    # Split the data into training and test sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Build the predictive model using RandomForest
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Predict on the training set and calculate accuracy
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    st.write(f'Model Accuracy on Training Data: {train_accuracy * 100:.2f}%')

    # Step 4: Show Model Performance (Confusion Matrix)
    cm_train = confusion_matrix(y_train, y_train_pred)
    
    # Improved visualization with better color contrast
    fig, ax = plt.subplots(figsize=(6, 5))  # Adjust figure size for better visibility
    cax = ax.imshow(cm_train, interpolation='nearest', cmap='coolwarm')  # Changed colormap to 'coolwarm'
    
    # Add color bar for better clarity
    fig.colorbar(cax)

    ax.set_title('Confusion Matrix (Training Data)')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(['Survived', 'Died'])
    ax.set_yticklabels(['Survived', 'Died'])

    # Annotate confusion matrix with text and adjust text color for contrast
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm_train[i, j], ha="center", va="center", color="white", fontsize=14)

    st.pyplot(fig)

    # Step 5: Upload the Test Data
    uploaded_test_file = st.file_uploader("Upload the Test Dataset (CSV format)", type=["csv"])
    if uploaded_test_file is not None:
        test_data = pd.read_csv(uploaded_test_file)
        st.write("Test Dataset Preview:")
        st.write(test_data.head())

        # Pre-process the test data
        test_data = test_data.dropna()
        X_test_data = test_data[features]
        y_test_data = test_data['DEATH_EVENT']

        # Step 6: Show Test Performance
        y_test_pred = model.predict(X_test_data)
        test_accuracy = accuracy_score(y_test_data, y_test_pred)
        st.write(f'Model Accuracy on Test Data: {test_accuracy * 100:.2f}%')

        # Show confusion matrix for the test data
        cm_test = confusion_matrix(y_test_data, y_test_pred)
        fig, ax = plt.subplots(figsize=(6, 5))  # Adjust figure size for better visibility
        cax = ax.imshow(cm_test, interpolation='nearest', cmap='coolwarm')  # Changed colormap to 'coolwarm'

        # Add color bar for better clarity
        fig.colorbar(cax)

        ax.set_title('Confusion Matrix (Test Data)')
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(2))
        ax.set_yticks(np.arange(2))
        ax.set_xticklabels(['Survived', 'Died'])
        ax.set_yticklabels(['Survived', 'Died'])

        # Annotate confusion matrix with text and adjust text color for contrast
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm_test[i, j], ha="center", va="center", color="white", fontsize=14)

        st.pyplot(fig)
