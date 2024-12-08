import os, pandas as pd, numpy as np, streamlit as st
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureEngineering:

    def __init__(self):
        self.filename = "integrated_phishing_data.csv"
        self.filepath="../data/integrated/"
        self.path = os.path.join(self.filepath, self.filename)
        self.df = pd.read_csv(self.path)
        self.df = self.df.dropna()

        self.df['special_char_ratio'] = self.df['num_special_chars'] / self.df['url_length']
        self.X = self.df[['url_length', 'num_special_chars', 'normal_url_length', 'normal_num_special_chars',
          'num_subdomains', 'is_https', 'is_domain_ip', 'total_suspicious_keywords', 'url_entropy', 'special_char_ratio']]
        self.y = self.df['label_encoded']    

    def principal_component_analysis(self):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        st.subheader('Explained Variance by Principal Component')
        pca = PCA(n_components=0.95)
        X_pca = pca.fit_transform(X_scaled)
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_ * 100)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance (%)')
        plt.title('Explained Variance by Principal Component')        
        st.pyplot(plt)
        pca_loadings = pd.DataFrame(pca.components_, columns=self.X.columns, index=[f'PC{i+1}' for i in range(pca.n_components_)])
        st.write(pca_loadings)

    def __calc_mean_legitimate_subdomain(self):
        mean_value = self.df['num_subdomains'].mean()
        return mean_value

    def feature_creation(self):
        # URL Complexity Score
        self.df['url_complexity_score'] = (self.df['url_length'] * self.df['num_special_chars'] * self.df['url_entropy'])

        # Normalized Suspicion Index
        self.df['normalized_suspicion_index'] = self.df['total_suspicious_keywords'] / self.df['url_length']

        # Subdomain to Special Character Ratio
        avg_subdomain_count_for_legitimate_links = self.__calc_mean_legitimate_subdomain()
        # self.df['subdomain_special_char_ratio'] = self.df['num_subdomains'] / self.df['num_special_chars'] if self.df['num_special_chars'] == 0 else avg_subdomain_count_for_legitimate_links
        self.df['subdomain_special_char_ratio'] = self.df.apply( lambda row: avg_subdomain_count_for_legitimate_links if row['num_special_chars'] == 0 else row['num_subdomains'] / row['num_special_chars'], axis=1)

        # HTTPS Suspicion Score
        self.df['https_suspicion_score'] = self.df['is_https'] * self.df['total_suspicious_keywords']

        # URL Normality Score
        self.df['url_normality_score'] = self.df['normal_url_length'] * self.df['normal_num_special_chars']

        # Entropy per Character
        self.df['entropy_per_character'] = self.df['url_entropy'] / self.df['url_length']

        new_features = ['url_complexity_score', 'normalized_suspicion_index', 'subdomain_special_char_ratio', 'https_suspicion_score', 'url_normality_score', 'entropy_per_character']
        self.X = self.df[['url_length', 'num_special_chars', 'normal_url_length', 'normal_num_special_chars', 'num_subdomains', 'is_https', 'is_domain_ip', 'total_suspicious_keywords', 'url_entropy', 'special_char_ratio'] + new_features]        
        self.principal_component_analysis()

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

        st.subheader("Chi-squared Scores and P-values")
        st.write(feature_scores.sort_values('Chi-squared Score', ascending=False))


if __name__ == '__main__':
    
    st.title('Phishing Detection Feature Engineering and PCA Analysis')
    action = st.sidebar.selectbox("Select an action", ["Chi-Square Feature Analysis", "PCA Analysis", "Feature Creation"])
    fe = FeatureEngineering()

    if action == "Feature Creation":
        st.write("Creating features...")
        fe.feature_creation()

    elif action == "PCA Analysis":
        st.write("Running Principal Component Analysis...")
        fe.principal_component_analysis()

    elif action == "Chi-Square Feature Analysis":
        st.write("Performing Feature Analysis...")
        fe.feature_analysis()
    
    # fe.feature_analysis()
    # fe.principal_component_analysis()
        # Retain the following features for modeling or further analysis:
        # url_length, num_special_chars, normal_url_length, normal_num_special_chars, url_entropy, num_subdomains, is_https, total_suspicious_keywords
    # fe.feature_creation()