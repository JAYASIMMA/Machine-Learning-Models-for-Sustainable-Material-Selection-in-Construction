import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import seaborn as sns

# Load the model
model = joblib.load('sustainable_material_model.pkl')

# Streamlit app layout
st.title("Sustainable Material Selection in Construction")

# Input fields for user data
cost = st.number_input("Cost of Material", min_value=0.0, max_value=1500.0, value=150.0)
strength = st.number_input("Strength of Material (MPa)", min_value=0.0, max_value=200.0, value=30.0)
durability = st.number_input("Durability of Material (Years)", min_value=0, max_value=100, value=10)
material_type = st.selectbox("Material Type", ["Concrete", "Steel", "Wood"])
environmental_impact = st.selectbox("Environmental Impact", ["Low", "Medium", "High"])
recyclability = st.selectbox("Recyclability", ["Yes", "No"])

# Prepare input for prediction
input_data = pd.DataFrame({
    'Cost': [cost],
    'Strength': [strength],
    'Durability': [durability],
    'Material Type': [material_type],
    'Environmental Impact': [environmental_impact],
    'Recyclability': [recyclability]
})

# Make prediction
if st.button("Predict Sustainability Rating"):
    prediction = model.predict(input_data)
    st.success(f"The predicted Sustainability Rating is: {prediction[0]}")
# Data Visualization Section
st.header("Data Visualization")
data = pd.read_csv('sustainable_materials_dataset.csv')

# Bar chart for sustainability ratings
sustainability_counts = data['Sustainability Rating'].value_counts()
st.bar_chart(sustainability_counts)

# Scatter plot for cost vs strength
st.subheader("Cost vs Strength Scatter Plot")
st.scatter_chart(data[['Cost', 'Strength']])

# Plotly box plot
fig = px.box(data, x='Material Type', y='Cost', color='Sustainability Rating',
             title="Cost Distribution by Material Type and Sustainability Rating")
st.plotly_chart(fig)

# Confusion Matrix
st.header("Confusion Matrix")
if st.button("Show Confusion Matrix"):
    y_true = data['Sustainability Rating']  # Assuming you have true labels in the dataset
    y_pred = model.predict(data.drop('Sustainability Rating', axis=1))  # Predict on the full dataset
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    st.pyplot(plt)

# Feature Importance Graph
st.header("Feature Importance")
if st.button("Show Feature Importance"):
    feature_importances = model.named_steps['classifier'].feature_importances_

    # Get the correct feature names from the OneHotEncoder
    categorical_features = ['Material Type', 'Environmental Impact', 'Recyclability']  # List all categorical features
    onehot = model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot']
    
    # Get feature names for all categorical features
    onehot_feature_names = onehot.get_feature_names_out(categorical_features)

    # Concatenate the target feature name with onehot_feature_names
    features = np.concatenate([np.array(['Sustainability Rating']), onehot_feature_names])

    # Check lengths
    print("Length of features:", len(features))
    print("Length of feature importances:", len(feature_importances))

    # Ensure the lengths match
    if len(features) != len(feature_importances):
        st.error("Mismatch in lengths of features and feature importances.")
    else:
        importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        fig = px.bar(importance_df, x='Importance', y='Feature', title="Feature Importance", orientation='h')
        st.plotly_chart(fig)

# ROC Curve
st.header("ROC Curve")
if st.button("Show ROC Curve"):
    from sklearn.metrics import roc_auc_score

    # Ensure y_true is correctly defined
    y_true = data['Sustainability Rating']  # Assuming you have true labels in the dataset

    # Get predicted probabilities for all classes
    y_prob = model.predict_proba(data.drop('Sustainability Rating', axis=1))

    # Check the shape of y_prob
    print("Shape of predicted probabilities:", y_prob.shape)

    # Assuming 'High' is the positive class, adjust if necessary
    pos_label_index = model.classes_.tolist().index('High')  # Get the index of the positive class
    y_prob_positive = y_prob[:, pos_label_index]  # Get probabilities for the positive class

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob_positive, pos_label='High')  # Adjust pos_label based on your classes
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    st.pyplot(plt)

# Display the dataset
if st.checkbox("Show Dataset"):
    st.dataframe(data)