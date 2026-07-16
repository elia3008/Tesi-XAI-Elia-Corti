# XAI Benchmark — tesi triennale

Benchmark centrato sull'utente per metodi di Explainable AI (SHAP, DiCE, Anchors)
applicati a un classificatore Random Forest sul dataset Heart Disease UCI.

Autore: Elia Corti — Università degli Studi di Milano-Bicocca  
Relatore: Prof. Navid Nobani  
A.A. 2025/2026

---

## Struttura del progetto

```
xai-benchmark/
├── data/
│   └── heart-disease-UCI.csv # dataset originale (versione preprocessata Kaggle/UCI)
| 
├── src/
│   ├── RF_model.py           # Fase A: addestra un modello Random Forest e lo salva
│   └── explain.py            # Fase B: genera le spiegazioni XAI dal modello
├── notebooks/
│   ├── 01_eda.ipynb          # esplorazione dati: distribuzioni, correlazioni, statistiche
│   └── 02_spiegazioni_xai.ipynb  # SHAP, DiCE e Anchors sui pazienti selezionati
├── artifacts/                # modello e split salvati (rigenerabili)
├── requirements.txt          # dipendenze con versioni congelate
└── README.md
```

---

## Come riprodurre i risultati

### 1. Clona il repository e crea l'ambiente virtuale

```bash
git clone https://github.com/elia3008/Tesi-XAI.git
cd xai-benchmark

python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows (PowerShell)

pip install -r requirements.txt
```

### 2. Esegui la pipeline nell'ordine

```bash
python src/RF_model.py        # genera artifacts/model.joblib e split.joblib
python src/explain.py            # genera le spiegazioni XAI in artifacts/
```

### 3. Esplora i notebook

```bash
jupyter notebook
```

Apri i notebook nella cartella `notebooks/` nell'ordine:
- `01_eda.ipynb` — analisi esplorativa del dataset pulito
- `02_spiegazioni_xai.ipynb` — spiegazioni SHAP, DiCE e Anchors per i pazienti selezionati



---

## Note sulla riproducibilità

- Il modello e lo split train/test sono salvati in `artifacts/` tramite `joblib`.
  Tutti gli script successivi caricano questi artefatti: non viene mai riallenato
  un modello diverso.
- `random_state=123` è fissato in ogni punto che introduce casualità
  (train_test_split, RandomForestClassifier).
- Le versioni esatte delle librerie sono in `requirements.txt`.
  Numpy è fissato a 1.26.x per compatibilità con alibi 0.9.6 e shap 0.52.0.
- La cartella `artifacts/` non è presente (vedi `.gitignore`): va rigenerata
  eseguendo la pipeline nell'ordine indicato sopra.

---

## Dataset

Heart Disease UCI — versione con missing imputation (Kaggle/ZTM), 303 righe.  
Fonte originale: UCI Machine Learning Repository, Cleveland dataset (id=45).  
Target: 0 = assenza di malattia cardiaca, 1 = presenza.
