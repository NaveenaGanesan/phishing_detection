import os
import pandas as pd
from tabulate import tabulate
import numpy as np
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

class DataDistribution:

    def __init__(self):
        self.filename = "integrated_phishing_data.csv"
        self.filepath="../data/integrated/"
        self.path = os.path.join(self.filepath, self.filename)
        self.df = pd.read_csv(self.path)

    def total_data_distribution(self):
        rows, cols = self.df.shape
        st.write(f"Total Data Distribution:: rows: {rows}, cols: {cols}")

    def target_class_distribution(self, target_label: str, display_plot: bool = False):
        phishing_count = self.df[target_label].sum()
        legitimate_count = self.df.shape[0] - phishing_count
        rows, cols = self.df.shape
        st.write(f"### Total Data Distribution:: rows: {rows}, cols: {cols}")
        st.write(f"##### Target Class Distribution:: Legitimate: {legitimate_count}, Phishing: {phishing_count}")
        # class_counts = self.df[target_label].value_counts()
        class_percentages = self.df[target_label].value_counts(normalize=True) * 100

        labels = ["legitimate", "phishing"]
        plt.pie(class_percentages, labels=labels, autopct='%.2f%%')
        plt.title("Class Distribution")        
        # print(f"Class Percentage: {class_percentages}")
        st.pyplot(plt)

    def descriptive_stats(self):
        # mean = lambda col: self.df[col].mean()
        # median = lambda col: self.df[col].median()
        # mode = lambda col: self.df[col].mode()
        # std = lambda col: self.df[col].std()
        # range = lambda col: self.df[col].max() - self.df[col].min()

        # desc = [mean, median, mode, std, range]
        # cols = ['url_length', 'num_special_chars', 'url_entropy']

        # table = [["column", "mean", "median", "mode", "std", "range"]]
        # for col in cols:
        #     stats = [f"{col}"]
        #     for func in desc:
        #         stats.append(func(col))
        #     table.append(stats)

        # print(tabulate(table, headers="firstrow", tablefmt="grid"))
        cols = ['url_length', 'num_special_chars', 'url_entropy']
        stats_df = self.df[cols].describe()
        st.write("### Descriptive Statistics")
        st.dataframe(stats_df)

    def correlation_analysis(self, display_plot: bool = False):
        features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        features = [f for f in features if f not in ['label', 'label_encoded']]
        correlation_matrix = self.df[features].corr()        
        plt.figure(figsize=(8, 6))  # Adjust figure size
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Correlation Matrix Heatmap")
        st.pyplot(plt)        

    def feature_analysis(self, features = None):
        """
        Chi Square test to identify important features
        is_https (χ² ≈ 58,314, p ≈ 0) is by far the most significant feature. The extremely high chi-square score and effectively zero p-value indicate that whether a URL uses HTTPS has a very strong relationship with the target classification.
        total_suspicious_keywords (χ² ≈ 1,379, p ≈ 7.6e-302) is the second most important feature. The very high chi-square score and extremely low p-value suggest that the presence of suspicious keywords is strongly associated with the URL classification.
        num_subdomains (χ² ≈ 247, p ≈ 1.3e-55) shows moderate importance, indicating that the number of subdomains in a URL is a meaningful predictor.
        The following features show weaker but still statistically significant relationships (p < 0.05):

        normal_url_length (χ² ≈ 23.4, p ≈ 1.3e-6)
        url_length (χ² ≈ 13.6, p ≈ 2.2e-4)
        normal_num_special_chars (χ² ≈ 10.7, p ≈ 0.001)
        url_entropy (χ² ≈ 8.0, p ≈ 0.005)
        num_special_chars (χ² ≈ 4.9, p ≈ 0.027)        
        """        

        if not features:
            features = self.df.select_dtypes(include=[np.number]).columns.tolist()
            features = [f for f in features if f not in ['label', 'label_encoded']]
        self.df = self.df.dropna()
        X = self.df[features] 
        y = self.df['label_encoded']

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        chi_scores, p_value = chi2(X_scaled, y)
        feature_scores = pd.DataFrame({
            'Feature': features,
            'Chi-squared Score': chi_scores,
            'P-value': p_value
        }).sort_values('Chi-squared Score', ascending=False)

        st.write("Important Features(Chi-Square Test)\n")
        st.dataframe(feature_scores)

    def tld_analysis(self, display_plot: bool = False):
        act_freq = self.df['tld'].value_counts()
        freq = act_freq[:5]        
        plt.bar(freq.index, freq.values)
        plt.title("Top 5 TLD Frequencies")
        plt.xlabel("TLD")
        plt.ylabel("Frequency")
        st.pyplot(plt)

    def url_length_analysis(self, display_plot: bool = False):
        # legitimate_lengths = self.df[self.df["label_encoded"] == 0]["url_length"]
        # phishing_lengths = self.df[self.df["label_encoded"] == 1]["url_length"]
        threshold = self.df['url_length'].quantile(0.999)
        filtered_df = self.df[self.df['url_length'] <= threshold]

        all_data  = pd.DataFrame({'url_length': filtered_df['url_length'], 'Category': "All"})
        legitimate_data  = pd.DataFrame({'url_length': filtered_df[filtered_df["label_encoded"] == 0]["url_length"], 'Category': "Legitimate"})
        phishing_data  = pd.DataFrame({'url_length': filtered_df[filtered_df["label_encoded"] == 1]["url_length"], 'Category': "Phishing"})

        combined_data = pd.concat([all_data, legitimate_data, phishing_data], ignore_index=True)

        sns.boxplot(data=combined_data, x='Category', y='url_length', palette='Set2')

        plt.title('Comparison of URL Lengths: All, Legitimate, and Phishing', fontsize=14)
        plt.xlabel('Category', fontsize=12)
        plt.ylabel('URL Length', fontsize=12)
        
        # plt.grid(axis='y', linestyle='--', alpha=0.7)
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        
        # if display_plot: plt.show()        
        st.pyplot(plt)

    def printHead(self) -> None:
        print(self.df.head())

if __name__ == "__main__":
    st.title("Phishing Data Dashboard")
    data_distribution = DataDistribution()
    st.sidebar.title("Navigation")
    options = [        
        "Target Class Distribution",
        "Descriptive Statistics",
        "Correlation Analysis",
        "Feature Analysis",
        "TLD Analysis",
        "URL Length Analysis"
    ]
    choice = st.sidebar.selectbox("Select a function to display", options)

    # Render the selected function
    # if choice == "Total Data Distribution":
    #     data_distribution.total_data_distribution()

    if choice == "Target Class Distribution":
        target_label = st.sidebar.text_input("Enter Target Label (default: 'label_encoded')", "label_encoded")
        data_distribution.target_class_distribution(target_label)

    elif choice == "Descriptive Statistics":
        data_distribution.descriptive_stats()

    elif choice == "Correlation Analysis":
        data_distribution.correlation_analysis()

    elif choice == "Feature Analysis":
        data_distribution.feature_analysis()

    elif choice == "TLD Analysis":
        data_distribution.tld_analysis()

    elif choice == "URL Length Analysis":
        data_distribution.url_length_analysis()