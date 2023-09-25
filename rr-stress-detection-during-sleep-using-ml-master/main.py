# Import necessary modules
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier

# Load data function
@st.cache()
def load_data():
    """This function returns the preprocessed data"""
    # Load the Diabetes dataset into DataFrame.
    df = pd.read_csv('Stress.csv')
    # Rename the column names in the DataFrame.
    df.rename(columns={"t": "bt"}, inplace=True)
    # Perform feature and target split
    X = df[["sr", "rr", "bt", "lm", "bo", "rem", "sh", "hr"]]
    y = df['sl']
    return df, X, y

# Train model function
@st.cache()
def train_model(X, y):
    """This function trains the model and returns the model and model score"""
    # Create the model
    model = DecisionTreeClassifier(
        ccp_alpha=0.0, class_weight=None, criterion='entropy',
        max_depth=4, max_features=None, max_leaf_nodes=None,
        min_impurity_decrease=0.0, min_samples_leaf=1,
        min_samples_split=2, min_weight_fraction_leaf=0.0,
        random_state=42, splitter='best'
    )
    # Fit the data on model
    model.fit(X, y)
    # Get the model score
    score = model.score(X, y)
    return model, score

# Data visualization function
def visualize_data(df):
    st.subheader("Data Visualization")
    
    # Example 1: Scatter Plot
    st.subheader("Scatter Plot")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='sr', y='rr', hue='sl', ax=ax)
    st.pyplot(fig)

    # Example 2: Histogram
    st.subheader("Histogram of BT")
    fig, ax = plt.subplots()
    sns.histplot(data=df, x='bt', kde=True, ax=ax)
    st.pyplot(fig)

    # Example 3: Bar Plot
    st.subheader("Bar Plot of HR")
    fig, ax = plt.subplots()
    sns.barplot(data=df, x='sl', y='hr', ax=ax)
    st.pyplot(fig)

    # Example 4: Line Plot
    st.subheader("Line Plot of LM")
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x='lm', y='sl', ax=ax)
    st.pyplot(fig)

    # Additional Plots

    # Example 5: Pair Plot
    st.subheader("Pair Plot")
    pair_plot = sns.pairplot(df, hue='sl')
    st.pyplot(pair_plot)

    # Example 6: Box Plot
    st.subheader("Box Plot of RR by Stress Level")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='sl', y='rr', ax=ax)
    st.pyplot(fig)

    # Example 7: Count Plot
    st.subheader("Count Plot of Stress Levels")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='sl', ax=ax)
    st.pyplot(fig)

    # You can add more plots as needed

# Prediction function
def predict_stress_level(X, y, features):
    model, _ = train_model(X, y)
    prediction = model.predict(np.array(features).reshape(1, -1))
    return prediction[0]

def main():
    st.title("Stress Level Detection App")

    # Load the data
    df, X, y = load_data()

    # Display model score
    model, score = train_model(X, y)
    st.subheader("Model Score")
    st.write(f"Accuracy: {score:.2f}")

    # Data visualization
    visualize_data(df)

    # Prediction section
    st.subheader("Make a Prediction")
    features = st.text_input("Enter features (comma-separated):")
    if features:
        features = [float(f) for f in features.split(',')]
        prediction = predict_stress_level(X, y, features)
        st.write(f"Predicted Stress Level: {prediction}")

if __name__ == "__main__":
    main()
