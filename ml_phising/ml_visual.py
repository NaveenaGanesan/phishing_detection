import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Bar Chart for Accuracy
def plot_accuracy_bar(df):
    st.subheader("Bar Chart: Model Accuracy")
    plt.figure(figsize=(10, 6))
    plt.bar(df["model_type"], df["accuracy"], color="skyblue")
    plt.xticks(rotation=45, ha="right")
    plt.title("Accuracy Comparison Across Models")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    st.pyplot(plt)

# Precision, Recall, and F1-Score Plot
def plot_metrics(df, model):
    st.subheader(f"Precision, Recall, and F1-Score for {model}")
    # Extracting metrics from the classification report
    selected_model = df[df["model_type"] == model].iloc[0]
    report = selected_model["classification_report"]
    lines = report.split("\n")

    # Parse metrics
    data = []
    for line in lines[2:4]:  # Parsing only Class 0 and Class 1
        if line.strip():
            parts = line.split()
            data.append([parts[0], float(parts[1]), float(parts[2]), float(parts[3])])

    # Convert to DataFrame
    metrics_df = pd.DataFrame(data, columns=["Class", "Precision", "Recall", "F1-Score"])

    # Plot
    metrics_df.set_index("Class")[["Precision", "Recall", "F1-Score"]].plot(kind="bar", figsize=(10, 6), color=["blue", "orange", "green"])
    plt.title(f"Precision, Recall, and F1-Score for {model}")
    plt.ylabel("Scores")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    st.pyplot(plt)

# Main App
def main():
    st.title("Model Results Visualization")
    
    file_path = st.text_input("Enter the full file path", "/Users/Dell/E/Code/phishing_detection/ml_phising/model_results.csv")

    df = load_data(file_path)

    # Show bar chart for accuracy
    plot_accuracy_bar(df)

    # Select model for detailed metrics
    model = st.selectbox("Select a model to view detailed metrics:", df["model_type"].unique())
    plot_metrics(df, model)

if __name__ == "__main__":
    main()
