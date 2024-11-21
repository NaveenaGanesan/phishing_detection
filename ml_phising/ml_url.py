import os, numpy, pandas as pd, lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

# Logistic regression
# Decision Trees
# Random Forest - (Hyperparameter Tuning - {GridSearchCV, RandomizedSearchCV}, More Feature Engg - {Textual Patterns, Domain Age}, Ensemble methods = {Gradient Boosting, XGBoost})
# SVM
# Gradient Boosting Machines

# XGBoost
# LightGBM
# Neural Networks
# KNN
# Ensemble Models

# cols:
# url,label,label_encoded,url_length,num_special_chars,normal_url_length,normal_num_special_chars,num_subdomains,is_https,is_domain_ip,total_suspicious_keywords,tld,url_entropy, special_char_ratio

def evaluation(y_test, y_pred, y_prob=None):
    # Evaluate accuracy and other metrics
    # accuracy = accuracy_score(y_test, y_pred)
    # report = classification_report(y_test, y_pred)
    # print("Accuracy:", accuracy)
    # print("Classification Report:\n", report)
    # res = {"accuracy": accuracy, "classification_report": report}

    # # ROC AUC (Area Under the Curve)
    # if y_prob is not None:
    #     auc_score = roc_auc_score(y_test, y_prob[:, 1])
    #     res['roc_auc'] = auc_score

    #     # Precision-Recall AUC
    #     precision, recall, _ = precision_recall_curve(y_test, y_prob[:, 1])
    #     pr_auc = auc(recall, precision)
    #     res['pr_auc'] = pr_auc
    res = {"y_test": y_test, "y_pred": y_pred, "y_prob": y_prob}
    return res

def save_results(model_name, parameters, metrics, file_path="model_results.csv"):
    # TODO:
    results = {
        'model_type': model_name,
        'parameters': parameters,
        'y_test': metrics['y_test'],
        'y_pred': metrics['y_pred'],
        'y_prob': metrics['y_prob'],
        # 'accuracy': metrics["accuracy"],
        # "classification_report": str(metrics['classification_report'])
    }

    results_df = pd.DataFrame([results])
    if os.path.isfile(file_path):
        existing_df = pd.read_csv(file_path)
        updated_df = pd.concat([existing_df, results_df], ignore_index=True)
    else:
        updated_df = results_df
    updated_df.to_csv(file_path, index=False)
    print(f"Results saved for {model_name} with parameters {parameters}")

def smote_fit(X, Y):
    smote = SMOTE(random_state=42)
    x, y = smote.fit_resample(X, Y)
    return x, y


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
    pca = PCA(n_components=10, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    model = SVC(kernel='linear', class_weight='balanced', random_state=42)
    # model = LinearSVC(class_weight='balanced', random_state=42)
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)
    return evaluation(y_test, y_pred), None

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
        xgb_model = XGBClassifier(random_state=42, use_label_encode=False)
        grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print("Best Parameters:", grid_search.best_params_)
        print("Best Cross-validation Score:", grid_search.best_score_)
        return best_params


    best_params = HyperParameterTuning_gridsearch()


    # model = XGBClassifier(
    #     n_estimators=200,        # Number of boosting rounds
    #     learning_rate=0.05,      # Step size shrinkage
    #     max_depth=5,             # Maximum tree depth
    #     random_state=42,         # Random seed for reproducibility
    #     use_label_encoder=False, # Disable automatic label encoder for warnings
    #     eval_metric='logloss'    # Evaluation metric to reduce warnings
    # )
    model = XGBClassifier(
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        subsample=best_params['subsample'],
        colsample=best_params['colsample_bytree'],
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    return evaluation(y_test, y_pred, y_proba), best_params

def lightgbm_model(X_train, X_test, y_train, y_test):

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    params = {
        'objective': 'binary',        # Assuming binary classification
        'metric': 'binary_error',     # Binary classification error metric
        'boosting_type': 'gbdt',      # Gradient Boosting Decision Tree
        'num_leaves': 31,             # Default value, can be tuned later
        'learning_rate': 0.1,         # From XGBoost's best params
        'feature_fraction': 1.0,      # From XGBoost's colsample_bytree
        'max_depth': 7,               # From XGBoost's best params
        'subsample': 0.8,             # From XGBoost's best params
        'verbose': -1,                # Suppress LightGBM logs
        # 'scale_pos_weight': 1.0
    }

    # Train the model with manual early stopping
    num_boost_round = 300
    best_score = numpy.inf
    best_iteration = 0
    patience = 50
    for i in range(num_boost_round):
        clf = lgb.train(params,
                        train_data,
                        num_boost_round=1,  # Train one round at a time
                        valid_sets=[test_data])

        # Evaluate the model on validation set (here we use the binary error metric)
        score = clf.best_score['valid_0']['binary_error']

        # Check if early stopping condition is met
        if score < best_score:
            best_score = score
            best_iteration = i
        else:
            if i - best_iteration >= patience:
                print(f"Early stopping at iteration {i}")
                break

    # Final model with best iteration
    clf = lgb.train(params,
                    train_data,
                    num_boost_round=best_iteration + 1,
                    valid_sets=[test_data])

    y_pred = clf.predict(X_test, num_iteration=best_iteration + 1)
    y_pred_binary = (y_pred >= 0.5).astype(int)
    return evaluation(y_test, y_pred_binary), params

def knn_model(X_train, X_test, y_train, y_test, n_neighbors = 5):
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
    # knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    # knn.fit(X_train_scaled, y_train)
    # y_pred = knn.predict(X_test_scaled)
    best_knn, best_params = HyperParameterTuning_GridSearchCV(X_train_scaled, y_train)
    best_knn.fit(X_train_scaled, y_train)
    y_pred = best_knn.predict(X_test_scaled)
    y_proba = best_knn.predict_proba(X_test_scaled)
    return evaluation(y_test, y_pred, y_proba), best_params


def run_models(df: pd.DataFrame):
    X = df[['url_length', 'num_special_chars', 'normal_url_length', 'normal_num_special_chars',
          'num_subdomains', 'is_https', 'is_domain_ip', 'total_suspicious_keywords', 'url_entropy', 'special_char_ratio']]
    y = df['label_encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_resampled, y_train_resampled = smote_fit(X_train, y_train)

    models = {
        "Logistic Regression": (logistic_regression, X_train_resampled, X_test, y_train_resampled, y_test, {'max_iter': 1000, 'class_weight': 'balanced'}, {"store": True, "only_execute": False}),
        "Random Forest": (random_forest, X_train_resampled, X_test, y_train_resampled, y_test, {'n_estimators': 100, 'random_state': 42, 'class_weight': 'balanced'}, {"store": False, "only_execute": False}),
        "Decision Tree": (decision_tree, X_train_resampled, X_test, y_train_resampled, y_test, {'max_depth': 10, 'random_state': 42, 'class_weight': 'balanced'}, {"store": False, "only_execute": False}),
        "SVM": (svm_model, X_train_resampled, X_test, y_train_resampled, y_test, {'kernel': 'linear', 'class_weight': 'balanced', 'random_state': 42}, {"store": False, "only_execute": False}),
        "SVM_PCA": (svm_model, X_train_resampled, X_test, y_train_resampled, y_test, {'kernel': 'linear', 'class_weight': 'balanced', 'random_state': 42, 'optimization':'pca-ncomponents=10'}, {"store": False, "only_execute": False}),
        "Gradient Boosting": (gradient_boosting, X_train_resampled, X_test, y_train_resampled, y_test, {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, "random_state": 42}, {"store": False, "only_execute": False}),
        "Gradient Boosting 1st iter": (gradient_boosting, X_train_resampled, X_test, y_train_resampled, y_test, {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 5, "random_state": 42}, {"store": False, "only_execute": False}),
        "Gradient Boosting subsample":(gradient_boosting, X_train_resampled, X_test, y_train_resampled, y_test, {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 5, 'random_state': 42, 'subsample': 0.9}, {"store": False, "only_execute": False}),
        "XGBoost": (xgboost_model, X_train_resampled, X_test, y_train_resampled, y_test, {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 5, "random_state": 42}, {"store": False, "only_execute": False}),
        "XGBoost Hypertuned GridSearchCV": (xgboost_model, X_train_resampled, X_test, y_train_resampled, y_test, {}, {"store": False, "only_execute": False}),
        "LightGBM Hypertuned GridSearchCV": (lightgbm_model, X_train_resampled, X_test, y_train_resampled, y_test, {}, {"store": False, "only_execute": False}),
        "KNN Neighbors": (knn_model, X_train_resampled, X_test, y_train_resampled, y_test, {'n_neighbors': 5}, {"store": False, "only_execute": False}),
        "KNN Neighbors HyperparameterTuned GridSearchCV": (knn_model, X_train_resampled, X_test, y_train_resampled, y_test, {}, {"store": False, "only_execute": False}),
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

    # logistic_regression(df.copy())
    # decision_tree(df.copy())
    # random_forest(df.copy())
    run_models(df)