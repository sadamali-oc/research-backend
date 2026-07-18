from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pickle
import os


def train_performance_model(df):

    print("\n===== MODEL TRAINING =====")

    # Features
    X = df.drop(
        columns=[
            "Performance Category",
            "Performance Score"
        ]
    )


    # Target
    y = df["Performance Category"]


    print("\nTarget Distribution:")
    print(y.value_counts())


    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    # Handle imbalance
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )


    weights = dict(
        zip(
            np.unique(y_train),
            class_weights
        )
    )


    # Random Forest Model
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight=weights
    )


    model.fit(
        X_train,
        y_train
    )


    # Prediction

    predictions = model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    print("\n===== MODEL PERFORMANCE =====")

    print(
        "Accuracy:",
        round(accuracy*100,2),
        "%"
    )


    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions
        )
    )


    # Save model

    os.makedirs(
        "models",
        exist_ok=True
    )


    with open(
        "models/performance_model.pkl",
        "wb"
    ) as file:

        pickle.dump(
            model,
            file
        )


    print(
        "\nModel saved at: models/performance_model.pkl"
    )


    return model