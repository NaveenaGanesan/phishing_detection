import os
import argparse
import pandas as pd
import numpy as np
import ipaddress
from collections import Counter
from urllib.parse import urlparse
from sklearn.preprocessing import LabelEncoder, StandardScaler

def label_encoding(label):    
    # Encode Module
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df[label])    

def feature_extraction(url):
    # Feature Extraction
    # Adding url length
    df['url_length'] = df[url].apply(len)

    # Special Character Count    
    df['num_special_chars'] = df[url].apply(lambda x: sum([1 for char in x if char in '-_?=&']))

def normalization(url_length = 'url_length', num_special_chars = 'num_special_chars') -> None:
    scaler = StandardScaler()
    old_norm_list = [url_length, num_special_chars]
    normalize_list = [f"normal_{url_length}", f"normal_{num_special_chars}"]
    df[normalize_list] = scaler.fit_transform(df[old_norm_list])

def no_of_subdomains(url = 'urls'):
    def get_total_subdomains(string):
        total = np.nan
        try:
            parser = urlparse(string)
            total = max(parser.netloc.count('.') - 1, 0)
        except Exception as e: pass            
        return total
    
    df["num_subdomains"] = df[url].apply(get_total_subdomains)    
    # print("Total Unique Subdomains:", df["num_subdomains"].unique())

def is_https(url = 'url'):
    def identify_if_https(string):
        return 1 if string.startswith('https') else 0
        
    df["is_https"] = df[url].apply(identify_if_https)

def is_domain_ip(url = 'url'):
    def domain_is_ip(string):
        try:
            ipaddress.ip_address(string)
            return 1
        except Exception as e:
            return 0
    df['is_domain_ip'] = df[url].apply(domain_is_ip)

def has_suspicious_keywords(url = 'url'):
    def suspicious_keywords():
        sus_keywords = ['login', 'verify', 'account', 'update', 'secure']
        
        ...
    df['total_suspicious_keywords'] = df[url].apply(suspicious_keywords)

def extract_tld(url = 'url'):
    def tld_extractor(string):
        tld = np.nan
        if not string.startswith(("http://", "https://")): string = "".join(["http://", string])
        try:
            parser = urlparse(string)
            lst = parser.netloc.split(".")
            if len(lst) > 1: tld = lst[-1]
        except ValueError:
            pass
        return tld

    df["tld"] = df[url].apply(tld_extractor)

def entropy(url = 'url'):
    def calculate_entropy(string):
        d = Counter(string)
        n = len(string)
        total = 0
        for c in string:
            prob = float(d[c]) / n
            total += prob * np.log2(prob)        
        return -total

    df['url_entropy'] = df[url].apply(calculate_entropy)    

def store_to_preprocess_csv(raw_data_filename, index = False) -> None:
    filename = raw_data_filename.split('.')[0]
    df.to_csv(f"../data/preprocess/{filename}_preprocess.csv", index = False)    


# Dataset details
dataset = {
    "1": {"filename": "kaggle_balanced_urls.csv", "label_encode_on": "label", "feature_extraction_on": "url"},    
    "2": {"filename": "kaggle_malicious_phish.csv", "label_encode_on": "type", "feature_extraction_on": "url"}, 
    "3": {"filename": "PhiUSIIL_Phishing_URL_Dataset_uci_2024.csv", "label_encode_on": "label", "feature_extraction_on": "url"},     
}

dataset_details = dataset["2"]

# Filename
raw_data_filename = dataset_details["filename"]
raw_data_filepath = '../data'
filepath = os.path.join(raw_data_filepath, raw_data_filename)

# Reading dataset
df = pd.read_csv(filepath)
df.columns = df.columns.str.lower()
parser = argparse.ArgumentParser(description="ETL script for preprocessing")
parser.add_argument("--preprocess", type=bool, default=False, help="Flag to whether preprocess data or not")
parser.add_argument("--store_preprocess", type=bool, default=False, help="Flag whether to store preprocess data or not")
args = parser.parse_args()

# Preprocessing functions
if args.preprocess:
    # Labelling nominal / ordinal variables
    label_encoding(dataset_details["label_encode_on"])

    # Calulating link length and special characters
    feature_extraction(dataset_details["feature_extraction_on"])

    # Normalizing url_data and special characters count
    normalization()

    no_of_subdomains(dataset_details["feature_extraction_on"])

    # Check if url is https or not
    is_https(dataset_details["feature_extraction_on"])

    is_domain_ip(dataset_details["feature_extraction_on"])

    extract_tld(dataset_details["feature_extraction_on"])

    # Calculating randomness of url
    entropy(dataset_details["feature_extraction_on"])


    null_validation = df.isnull().sum()
    print(type(null_validation), null_validation)


# Printing new data
print(df.columns)
print(df.head())


if args.store_preprocess:
    # Save the preprocessed DataFrame
    store_to_preprocess_csv(raw_data_filename)