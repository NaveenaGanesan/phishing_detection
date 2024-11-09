import os
import argparse
import pandas as pd
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

def store_to_preprocess_csv(raw_data_filename, index = False) -> None:
    filename = raw_data_filename.split('.')[0]
    df.to_csv(f"../data/preprocess/{filename}_preprocess.csv", index = False)    


# Dataset details
dataset = {
    "1": {"filename": "kaggle_balanced_urls.csv", "label_encode_on": "label", "feature_extraction_on": "url"},    
    "2": {"filename": "kaggle_malicious_phish.csv", "label_encode_on": "type", "feature_extraction_on": "url"},           
}

dataset_details = dataset["2"]

# Filename
raw_data_filename = dataset_details["filename"]
raw_data_filepath = '../data'
filepath = os.path.join(raw_data_filepath, raw_data_filename)

# Reading dataset
df = pd.read_csv(filepath)

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

    null_validation = df.isnull().sum()
    print(type(null_validation), null_validation)


# Printing new data
print(df.head())

if args.store_preprocess:
    # Save the preprocessed DataFrame
    store_to_preprocess_csv(raw_data_filename)