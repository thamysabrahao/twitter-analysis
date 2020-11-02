from Services.model_functions import *
from Services.data_cleansing import *
import nltk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
nltk.download('stopwords')


def train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=42,
                                                        test_size=0.2)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    df = read_csv('Train100.csv.zip')
    df = df.sample(frac=1).reset_index(drop=True)

    # Data cleansing
    df['normalized_text'] = df['tweet_text'].apply(lambda column: string_normalize(column))
    df['clean_text'] = df['normalized_text'].apply(valid_text)

    # Model training and hyperparameter optimization
    X = list(df['clean_text'].astype(str))
    y = df['sentiment']

    # Modelling
    X_train, X_test, y_train, y_test = train_test(X, y)

    print('Vectorizing')
    count_vectorizer = CountVectorizer()
    X_train_count = count_vectorizer.fit_transform(X_train)
    X_test_count = count_vectorizer.transform(X_test)
    save_model(count_vectorizer, "count_vectorizer")

    print('LogisticRegression')
    log = LogisticRegression(random_state=42, solver='saga')
    log_reg = log.fit(X_train_count, y_train).predict(X_test_count)
    save_model(log, "log")

    # Baseline model
    print("Baseline model result:")
    print(classification_report(y_test, log_reg, target_names=['negative', 'positive']))

    print('MultinomialNB')
    cl_multi = MultinomialNB()
    res_multi = cl_multi.fit(X_train_count, y_train).predict(X_test_count)
    print("MultinomialNB model result:")
    print(classification_report(y_test, res_multi, target_names=['negative', 'positive']))
    save_model(cl_multi, "cl_multi")

    print('Random Forest')
    rf = RandomForestClassifier(random_state=42)
    rf_pred = rf.fit(X_train_count, y_train)
    predictions_rf = rf_pred.predict(X_test_count)
    print("Random Forest model result:")
    print(classification_report(y_test, predictions_rf, target_names=['negative', 'positive']))
    save_model(rf, "random_forest")

    print('Optimizing hyperparameters')
    model = RandomForestClassifier(random_state=42)

    # Grid search
    grid = {"n_estimators": [10, 50, 100],
            "criterion": ["gini", "entropy"],
            "bootstrap": [True, False],
            "max_features": ["auto", "sqrt", "log2"]
            }

    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='precision', error_score=0)
    grid_result = grid_search.fit(X_train_count, y_train)

    best_model = RandomForestClassifier(n_estimators=grid_result.best_params_["n_estimators"],
                                        criterion=grid_result.best_params_["criterion"],
                                        bootstrap=grid_result.best_params_["bootstrap"],
                                        max_features=grid_result.best_params_["max_features"],
                                        random_state=42)
    best_model.fit(X_train_count, y_train)
    y_pred = best_model.predict(X_test_count)
    print("Optimizaded Random Forest model result:")
    print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))
    save_model(best_model, "random_forest_opt_model")

    print('Ensemble: Voting Classifier')
    ml_hard = VotingClassifier(estimators=[('MultinomialNB', cl_multi),
                                           ('RF', rf),
                                           ('RF_opt', best_model)],
                               voting='hard').fit(X_train_count, y_train)

    y_pred_hard = ml_hard.predict(X_test_count)
    save_model(ml_hard, "ensemble_model")
    print('Ensemble model result:')
    print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))
    print('***********************')







