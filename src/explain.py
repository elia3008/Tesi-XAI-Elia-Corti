"""Fase B - generazione delle spiegazioni XAI SHAP del paziente 0.
"""
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

ART = Path("artifacts")


def main():
    model = joblib.load(ART / "model.joblib")
    split = joblib.load(ART / "split.joblib")
    X_test = split["X_test"]

    explainer = shap.TreeExplainer(model)
    expl = explainer(X_test)[:, :, 1]   # classe 1 = malattia

    plt.figure()
    shap.plots.waterfall(expl[0], show=False, max_display=10)
    plt.tight_layout()
    out = ART / "shap_paziente_0.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Spiegazione locale salvata: {out}")


if __name__ == "__main__":
    main()
