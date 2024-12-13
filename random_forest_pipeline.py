import argparse
import os
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

def main():
    """Main execution workflow."""
    parser = argparse.ArgumentParser(description="Train a Random Forest model on the HAR dataset.")
    parser.add_argument("--training_data", type=str, required=True, help="Path to the dataset (CSV file).")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in the forest.")
    parser.add_argument("--max_depth", type=int, default=None, help="Maximum depth of the trees.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of the data to use as test set.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--k_features", type=int, default=10, help="Number of top features to select.")
    args = parser.parse_args()

    if not os.path.exists(args.training_data):
        raise FileNotFoundError(f"The dataset file {args.training_data} does not exist.")

    print(f"Loading data from {args.training_data}...")
    data = pd.read_csv(args.training_data)

    X = data.iloc[:, :-1]  # All columns except the last
    y = data.iloc[:, -1]   # The last column is the target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    X_train_reduced, X_test_reduced, selected_features = feature_selection(
        X_train, y_train, X_test, k=args.k_features
    )

    # Start MLflow run
    mlflow.start_run()
    try:
        mlflow.sklearn.autolog()  # Enable autologging for sklearn

        model = train_random_forest(
            X_train_reduced, y_train, X_test_reduced, y_test,
            args.n_estimators, args.max_depth, args.random_state
        )

        # Log selected features explicitly
        mlflow.log_param("selected_features", list(selected_features))

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        mlflow.end_run()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
