"""Fase A - training e tuning modello Random Forest
"""
from pathlib import Path
import numpy as np
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score

DATA = Path("data/heart-disease-UCI.csv")
ART = Path("artifacts")
RANDOM_STATE = 123

df = pd.read_csv(r"C:\Users\ELIA\Documents\Tesi\Script\Tesi-XAI\data\heart_disease_cleveland.csv")
print(f"Shape: {df.shape}")
print(f"\nTarget distribution:\n{df['target'].value_counts()}")
print(f"\nDisease rate: {df['target'].mean():.2%}")
df.head()


rf_grid = {
    "n_estimators":      np.arange(100, 500, 50),      
    "max_depth":         [None, 3, 5, 8, 10],          
    "min_samples_split": np.arange(2, 20, 2),          
    "min_samples_leaf":  np.arange(1, 20, 2),          
    "max_features":      ["sqrt", "log2", None],
}


def main():
    # Setup random seed
    np.random.seed(RANDOM_STATE)

    #df = pd.read_csv(DATA)
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Ricerca casuale degli iperparametri
    rs_rf = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_distributions=rf_grid,
        cv=5,
        n_iter=20,
        scoring="roc_auc",
        verbose=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rs_rf.fit(X_train, y_train)

    print("\nMigliori iperparametri trovati:")
    for k, v in rs_rf.best_params_.items():
        print(f"  {k:20}: {v}")
    print(f"\nAUC media in cross-validation (tuning): {rs_rf.best_score_:.3f}")

    # Confronto tunato vs default sul test set 
    tuned = rs_rf.best_estimator_
    default = RandomForestClassifier(random_state=RANDOM_STATE).fit(X_train, y_train)

    print("\nConfronto sul test set:")
    for nome, mdl in [("Default", default), ("Tunato", tuned)]:
        pred = mdl.predict(X_test)
        proba = mdl.predict_proba(X_test)[:, 1]
        print(f"  {nome:8}  Accuracy={accuracy_score(y_test, pred):.3f}  "
              f"AUC={roc_auc_score(y_test, proba):.3f}")

    # Tuning non migliora: uso il modello default
    ART.mkdir(exist_ok=True)
    joblib.dump(default, ART / "model.joblib")
    joblib.dump(
        {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test},
        ART / "split.joblib",
    )
    print(f"\nArtefatti salvati in {ART}/ (model.joblib = modello default, split.joblib)")


if __name__ == "__main__":
    main()