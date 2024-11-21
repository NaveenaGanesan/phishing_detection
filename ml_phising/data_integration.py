import os
import pandas as pd
import argparse

def read_all_preprocess(dataset_obj: dict) -> list:
    arr = []
    for obj in dataset_obj.values():
        try:
            file = os.path.join(obj["file_path"], obj["filename"])
            df = pd.read_csv(file)
            arr.append(df)
        except Exception as e:
            print(f"Error on {obj['filename']}: {e}")
    return arr

def merge_all_preprocess(preprocess_list: list) -> pd.DataFrame:
    try:
        merged_df = pd.concat(preprocess_list, axis=0, ignore_index=True)
        return merged_df
    except Exception as e:
        print("Error on merging all files:", e)
    return None

def final_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # df = df.fillna(method='ffill')
    # df = df.fillna(0)

    # Drop duplicates
    df = df.drop_duplicates(subset='url', keep='first')

    # Drop any other unnecessary columns
    cols_to_keep = ["url","label","label_encoded","url_length","num_special_chars","normal_url_length","normal_num_special_chars","num_subdomains","is_https","is_domain_ip","total_suspicious_keywords","tld","url_entropy"]
    cols_to_drop = [col for col in df.columns if col not in cols_to_keep]
    df = df.drop(columns=cols_to_drop)

    # print("All uniques values in label column")


    return df

def store_df(df: pd.DataFrame, directory = '../data/integrated/', filename = 'integrated_phishing_data.csv') -> None:
    os.makedirs(directory, exist_ok=True)
    final_path = os.path.join(directory, filename)
    df.to_csv(final_path, index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Argument Parser for Data Integration")
    parser.add_argument("--store", type=bool, default=False, help="Flag whether to store merged data or not!")
    args = parser.parse_args()


    preprocess_list = []
    raw_file_path = "../data/preprocess/"

    dataset_details_obj = {
        "1": {"filename": "all_url_preprocess.csv", "file_path": raw_file_path},
        "2": {"filename": "kaggle_malicious_phish_preprocess.csv", "file_path": raw_file_path},
        "3": {"filename": "kaggle_balanced_urls_preprocess.csv", "file_path": raw_file_path},
        "4": {"filename": "PhiUSIIL_Phishing_URL_Dataset_uci_2024_preprocess.csv", "file_path": raw_file_path},
    }
    preprocess_list = read_all_preprocess(dataset_details_obj)
    final_df = merge_all_preprocess(preprocess_list)
    final_df = final_preprocess(final_df)
    print(final_df.head())
    if args.store: store_df(final_df)