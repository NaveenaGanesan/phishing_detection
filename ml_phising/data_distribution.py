import os
import pandas as pd
from tabulate import tabulate
import numpy as np
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

class DataDistribution:

    def __init__(self):
        self.filename = "integrated_phishing_data.csv"
        self.filepath="../data/integrated/"
        self.path = os.path.join(self.filepath, self.filename)
        self.df = pd.read_csv(self.path)

    def total_data_distribution(self):
        rows, cols = self.df.shape
        print(f"Total Data Distribution:: rows: {rows}, cols: {cols}")

    def target_class_distribution(self, target_label: str):
        phishing_count = self.df[target_label].sum()
        legitimate_count = self.df.shape[0] - phishing_count
        print(f"Target Class Distribution:: Legitimate: {legitimate_count}, Phishing: {phishing_count}")
        class_counts = self.df[target_label].value_counts()
        class_percentages = self.df[target_label].value_counts(normalize=True) * 100

        labels = ["legitimate", "phishing"]
        plt.pie(class_percentages, labels=labels, autopct='%.2f%%')
        plt.show()
        print(f"Class Percentage: {class_percentages}")

    def descriptive_stats(self):
        mean = lambda col: self.df[col].mean()
        median = lambda col: self.df[col].median()
        mode = lambda col: self.df[col].mode()
        std = lambda col: self.df[col].std()
        range = lambda col: self.df[col].max() - self.df[col].min()

        desc = [mean, median, mode, std, range]
        cols = ['url_length', 'num_special_chars', 'url_entropy']

        table = [["column", "mean", "median", "mode", "std", "range"]]
        for col in cols:
            stats = [f"{col}"]
            for func in desc:
                stats.append(func(col))
            table.append(stats)

        print(tabulate(table, headers="firstrow", tablefmt="grid"))

    def correlation_analysis(self):
        features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        features = [f for f in features if f not in ['label', 'label_encoded']]
        correlation_matrix = self.df[features].corr()
        # print(correlation_matrix)
        plt.figure(figsize=(8, 6))  # Adjust figure size
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')

        plt.title("Correlation Matrix Heatmap")
        # plt.show()

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

        self.df = self.df.dropna()

        if not features:
            features = self.df.select_dtypes(include=[np.number]).columns.tolist()
            features = [f for f in features if f not in ['label', 'label_encoded']]

        X = self.df[features]
        y = self.df['label_encoded']

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        chi_scores, p_value = chi2(X_scaled, y)
        feature_scores = pd.DataFrame({
            'Feature': features,
            'Chi-squared Score': chi_scores,
            'P-value': p_value
        })

        print("Important Features...\n")
        print(feature_scores.sort_values('Chi-squared Score', ascending=False))

    def tld_analysis(self):
        act_freq = self.df['tld'].value_counts()
        freq = act_freq[:5]
        height = np.arange(5)
        plt.bar(freq.index, freq.values)
        plt.show()         
        print(act_freq)        

    def url_length_analysis(self):
        ...

    def printHead(self) -> None:
        print(self.df.head())

if __name__ == "__main__":
    dd = DataDistribution()
    # dd.total_data_distribution()
    # dd.target_class_distribution('label_encoded')
    # dd.descriptive_stats()
    # dd.correlation_analysis()
    # dd.feature_analysis()
    dd.tld_analysis()
    # dd.printHead()