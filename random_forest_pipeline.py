import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns

def feature_selection(X_train, y_train, X_test, k=10):
    """Select top k features."""
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_reduced = selector.fit_transform(X_train, y_train)
    X_test_reduced = selector.transform(X_test)

    selected_feature_indices = selector.get_support(indices=True)
    selected_features = X_train.columns[selected_feature_indices]
    print("Selected Features:", list(selected_features))

    return X_train_reduced, X_test_reduced, selected_features

def train_random_forest(X_train, y_train, X_test, y_test, n_estimators, max_depth, random_state):
    """Train and evaluate a Random Forest model."""
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight='balanced',
        random_state=random_state
    )
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    classes = y_test.unique()
    conf_matrix = confusion_matrix(y_test, y_pred, labels=classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

    return rf_model

    # Get the arugments we need to avoid fixing the dataset path in code
parser = argparse.ArgumentParser()
parser.add_argument("--trainingdata", type=str, required=True, help='Dataset for training')
args = parser.parse_args()
mlflow.autolog()
df = pd.read_csv(args.trainingdata)
print(df)

    X = data.iloc[:, :-1]  # All columns except the last
    y = data.iloc[:, -1]   # The last column is the target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    X_train_reduced, X_test_reduced, selected_features = feature_selection(
        X_train, y_train, X_test, k=args.k_features
    )



