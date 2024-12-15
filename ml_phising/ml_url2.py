import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Evaluation function
# def evaluation(y_test, y_pred, y_prob=None):
#     res = {"y_test": y_test, "y_pred": y_pred, "y_prob": y_prob}
#     return res

def evaluation(y_test, y_pred, y_prob=None):
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    res = {
        "y_test": y_test.tolist(),  # Convert to list for JSON compatibility
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist() if y_prob is not None else None,
        "accuracy": accuracy,
        "classification_report": report,
    }
    return res

# Save results
# def save_results(model_name, parameters, metrics, file_path="model_results.csv"):
#     results = {
#         'model_type': model_name,
#         'parameters': parameters,
#         'y_test': metrics['y_test'],
#         'y_pred': metrics['y_pred'],
#         'y_prob': metrics['y_prob']
#     }
    
#     results_df = pd.DataFrame([results])
#     if os.path.isfile(file_path):
#         existing_df = pd.read_csv(file_path)
#         updated_df = pd.concat([existing_df, results_df], ignore_index=True)
#     else:
#         updated_df = results_df
#     updated_df.to_csv(file_path, index=False)
#     print(f"Results saved for {model_name} with parameters {parameters}")

def save_results(model_name, parameters, metrics, file_path="model_results.csv"):
    results = {
        'model_type': model_name,
        'parameters': parameters,
        'accuracy': metrics['accuracy'],
        'classification_report': metrics['classification_report'],
    }
    results_df = pd.DataFrame([results])
    if os.path.isfile(file_path):
        existing_df = pd.read_csv(file_path)
        updated_df = pd.concat([existing_df, results_df], ignore_index=True)
    else:
        updated_df = results_df
    updated_df.to_csv(file_path, index=False)
    print(f"Results saved for {model_name} with parameters {parameters}")

# SMOTE fitting
def smote_fit(X, Y):
    smote = SMOTE(random_state=42)
    x, y = smote.fit_resample(X, Y)
    return x, y

# Models
def logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    return evaluation(y_test, y_pred, y_proba), None

def random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    return evaluation(y_test, y_pred, y_proba), None

def decision_tree(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    return evaluation(y_test, y_pred, y_proba), None

def svm_model(X_train, X_test, y_train, y_test):
    model = SVC(kernel='linear', class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.decision_function(X_test)  # SVM does not have predict_proba by default
    return evaluation(y_test, y_pred, y_proba), None

def gradient_boosting(X_train, X_test, y_train, y_test):
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, subsample=0.9)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    return evaluation(y_test, y_pred, y_proba), None

def xgboost_model(X_train, X_test, y_train, y_test):
    def HyperParameterTuning_gridsearch():
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        xgb_model = XGBClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print("Best Parameters:", grid_search.best_params_)
        print("Best Cross-validation Score:", grid_search.best_score_)
        return best_params

    best_params = HyperParameterTuning_gridsearch()
    model = XGBClassifier(
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    return evaluation(y_test, y_pred, y_proba), best_params

def lightgbm_model(X_train, X_test, y_train, y_test):
    def HyperParameterTuning_gridsearch():
        param_grid = {
            'num_leaves': [31, 50, 70],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        lgb_model = lgb.LGBMClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=lgb_model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print("Best Parameters:", grid_search.best_params_)
        return best_params

    best_params = HyperParameterTuning_gridsearch()
    model = lgb.LGBMClassifier(
        num_leaves=best_params['num_leaves'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    return evaluation(y_test, y_pred, y_proba), best_params

def knn_model(X_train, X_test, y_train, y_test):
    def HyperParameterTuning_GridSearchCV(X_train_scaled, y_train):
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }
        knn = KNeighborsClassifier()
        grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train_scaled, y_train)
        best_params = grid_search.best_params_
        best_knn = grid_search.best_estimator_
        return best_knn, best_params

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    best_knn, best_params = HyperParameterTuning_GridSearchCV(X_train_scaled, y_train)
    best_knn.fit(X_train_scaled, y_train)
    y_pred = best_knn.predict(X_test_scaled)
    y_proba = best_knn.predict_proba(X_test_scaled)
    return evaluation(y_test, y_pred, y_proba), best_params

# Running models
def run_models(df: pd.DataFrame):
    # Select relevant features
    X = df[['url_length', 'num_special_chars', 'normal_url_length', 'normal_num_special_chars', 
            'url_entropy', 'num_subdomains', 'is_https', 'total_suspicious_keywords']]
    y = df['label_encoded']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply SMOTE
    X_train_resampled, y_train_resampled = smote_fit(X_train, y_train)

    models = {
        "Logistic Regression": (logistic_regression, X_train_resampled, X_test, y_train_resampled, y_test, {'max_iter': 1000, 'class_weight': 'balanced'}, {"store": False, "only_execute": False}),
        "Random Forest": (random_forest, X_train_resampled, X_test, y_train_resampled, y_test, {'n_estimators': 100, 'random_state': 42, 'class_weight': 'balanced'}, {"store": False, "only_execute": False}),
        "Decision Tree": (decision_tree, X_train_resampled, X_test, y_train_resampled, y_test, {'max_depth': 10, 'random_state': 42, 'class_weight': 'balanced'}, {"store": False, "only_execute": False}),
        "SVM": (svm_model, X_train_resampled, X_test, y_train_resampled, y_test, {'kernel': 'linear', 'class_weight': 'balanced', 'random_state': 42}, {"store": False, "only_execute": False}),
        "Gradient Boosting": (gradient_boosting, X_train_resampled, X_test, y_train_resampled, y_test, {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 5}, {"store": True, "only_execute": False}),        
        "XGBoost HyperparameterTuned": (xgboost_model, X_train_resampled, X_test, y_train_resampled, y_test, {}, {"store": True, "only_execute": False}),
        "LightGBM HyperparameterTuned": (lightgbm_model, X_train_resampled, X_test, y_train_resampled, y_test, {}, {"store": True, "only_execute": False}),
        "KNN HyperparameterTuned": (knn_model, X_train_resampled, X_test, y_train_resampled, y_test, {}, {"store": True, "only_execute": False}),

        # "XGBoost": (xgboost_model, X_train_resampled, X_test, y_train_resampled, y_test, {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 5}, {"store": True, "only_execute": False}),
        # "LightGBM": (lightgbm_model, X_train_resampled, X_test, y_train_resampled, y_test, {}, {"store": True, "only_execute": False}),
        # "KNN": (knn_model, X_train_resampled, X_test, y_train_resampled, y_test, {'n_neighbors': 5}, {"store": True, "only_execute": False}),          
    }

    for model, (model_func, X_train, X_test, y_train, y_test, params, obj) in models.items():
        if obj["only_execute"] or obj["store"]:
            print(f"Running {model}...")
            metrics, hyper_params = model_func(X_train, X_test, y_train, y_test)
        if obj["store"]: save_results(model, params if hyper_params == None else hyper_params, metrics)

if __name__ == "__main__":
    df = pd.read_csv("../data/integrated/integrated_phishing_data.csv", low_memory=False)
    df.dropna(inplace=True)
    df['special_char_ratio'] = df['num_special_chars'] / (df['url_length'] + 1)
    run_models(df)
