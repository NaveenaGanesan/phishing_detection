import os
import argparse
import pandas as pd
import numpy as np
import ipaddress
import re
from collections import Counter
from urllib.parse import urlparse
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = None

def label_encoding(label):
    global df
    # Encode Module
    df[label] = df[label].astype(str).str.lower()

    # 1: is bad, 0: safe
    phishing_terms = ['phishing', 'malicious', 'malware', 'defacement', 'suspicious', 'fake', 0, "0"]

    df['label_encoded'] = df[label].apply(
        lambda x: 1 if any(str(term) in x for term in phishing_terms) else 0
    )    
    # label_encoder = LabelEncoder()
    # df['label_encoded'] = label_encoder.fit_transform(df[label])
    df = df.rename(columns={label: "label"})


def normal_url_length(url):
    global df
    # Feature Extraction
    # Adding url length
    df['url_length'] = df[url].apply(len)

    # Special Character Count
    df['num_special_chars'] = df[url].apply(lambda x: sum([1 for char in x if char in '-_?=&']))

def normalization(url_length = 'url_length', num_special_chars = 'num_special_chars') -> None:
    global df
    scaler = StandardScaler()
    old_norm_list = [url_length, num_special_chars]
    normalize_list = [f"normal_{url_length}", f"normal_{num_special_chars}"]
    df[normalize_list] = scaler.fit_transform(df[old_norm_list])

def no_of_subdomains(url = 'urls'):
    global df
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
    global df
    def identify_if_https(string):
        return 1 if string.startswith('https') else 0

    df["is_https"] = df[url].apply(identify_if_https)

def is_domain_ip(url = 'url'):
    global df
    def domain_is_ip(string):
        try:
            ipaddress.ip_address(string)
            return 1
        except Exception as e:
            return 0
    df['is_domain_ip'] = df[url].apply(domain_is_ip)

def has_suspicious_keywords(url = 'url'):
    global df
    # Security-related terms: "secure", "safety", "protection"
    # Action words: "verify", "confirm", "update", "login"
    # Urgency indicators: "immediate", "urgent", "important"
    # Financial terms: "account", "bank", "credit", "payment"
    # sus_keywords = ["secure", "safety", "protection", "verify", "confirm", "update", "login", "immediate", "urgent", "important", "account", "bank", "credit", "payment"]

    sus_keywords_pattern = re.compile(r'\b(secure|safety|protection|verify|confirm|update|login|immediate|urgent|important|account|bank|credit|payment)', re.IGNORECASE)
    def suspicious_keywords(string):
        matches = sus_keywords_pattern.findall(string)
        return len(matches)
    
    df['total_suspicious_keywords'] = df[url].apply(suspicious_keywords)

def extract_tld(url = 'url'):
    global df
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
    global df
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
    global df
    filename = raw_data_filename.split('.')[0]
    df.to_csv(f"../data/preprocess/{filename}_preprocess.csv", index = False)


# Dataset details
dataset = {
    "1": {"filename": "kaggle_balanced_urls.csv", "label_encode_on": "label", "feature_extraction_on": "url", "excess_columns": True},
    "2": {"filename": "kaggle_malicious_phish.csv", "label_encode_on": "type", "feature_extraction_on": "url"},
    "3": {"filename": "PhiUSIIL_Phishing_URL_Dataset_uci_2024.csv", "label_encode_on": "label", "feature_extraction_on": "url", "excess_columns": True},
    "4": {"filename": "all_url.csv", "label_encode_on": "type", "feature_extraction_on": "url"},    
}


def preprocess():
    global df
    if "excess_columns" in dataset_details: df = df[['url', 'label']]
    # Preprocessing functions
    if args.preprocess:
        # Labelling nominal / ordinal variables (label_encoded_on: numeric(Mostly 0, 1))
        label_encoding(dataset_details["label_encode_on"])

        # Calulating link length and special characters (url_length: numeric) (num_special_chars: numeric)
        normal_url_length(dataset_details["feature_extraction_on"])

        # Normalizing url_data and special characters count (normal_url_length: float) (normal_num_special_chars: float)
        normalization()

        # Count number of sub-domains (num_subdomains: numeric)
        no_of_subdomains(dataset_details["feature_extraction_on"])

        # Check if url is https or not (is_https: boolean)
        is_https(dataset_details["feature_extraction_on"])

        # Check for domain ip (Check if actually an url is ip address or domain name) (is_domain_ip: boolean)
        is_domain_ip(dataset_details["feature_extraction_on"])

        # Count total suspicious keywords (total_suspicious_keywords: numeric)
        has_suspicious_keywords(dataset_details["feature_extraction_on"])

        # Get Top-Level-Domain (tld: string)
        extract_tld(dataset_details["feature_extraction_on"])

        # Calculating randomness of url (Randomness in charcters in an url)
        entropy(dataset_details["feature_extraction_on"])

        # Are there null values
        null_validation = df.isnull().sum()
        print(type(null_validation), null_validation)

    # Printing new data
    print(df.columns)
    print(df.head())

    if args.store_preprocess:
        # Save the preprocessed DataFrame
        global raw_data_filename
        store_to_preprocess_csv(raw_data_filename)


parser = argparse.ArgumentParser(description="ETL script for preprocessing")
parser.add_argument("--preprocess", type=bool, default=False, help="Flag to whether preprocess data or not")
parser.add_argument("--store_preprocess", type=bool, default=False, help="Flag whether to store preprocess data or not")
parser.add_argument("--preprocess_all", type=bool, default=False, help="Flag to whether preprocess all datasets or not")
args = parser.parse_args()
raw_data_filename = None


def read_data(dataset_details):    
    global df, raw_data_filename
    # Filename
    raw_data_filename = dataset_details["filename"]
    raw_data_filepath = '../data/csvs/'
    if "excess_path" in dataset_details:
        raw_data_filepath = os.path.join(raw_data_filepath, dataset_details["excess_path"])
    filepath = os.path.join(raw_data_filepath, raw_data_filename)

    # Reading dataset
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower()


dataset_details = dataset["4"]
if args.preprocess_all:
    for key in dataset.keys():
        dataset_details = dataset[key]
        read_data(dataset_details)
        preprocess()
else:
    read_data(dataset_details)
    preprocess()
    # print(df['label'].unique())