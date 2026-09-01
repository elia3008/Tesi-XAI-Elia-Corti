# XAI Benchmark — tesi triennale

Benchmark centrato sull'utente per metodi di Explainable AI (SHAP, DiCE, Anchors)
applicati a un classificatore Random Forest sul dataset Heart Disease UCI.

Autore: Elia Corti — Università degli Studi di Milano-Bicocca  
Relatore: Prof. Navid Nobani  
A.A. 2025/2026

---

## Struttura del progetto

```
Tesi-XAI-Elia-Corti/
├── data/
│   └── heart_disease_cleveland.csv  # dataset Heart Disease Cleveland (UCI), codifica grezza
│
├── src/
│   ├── RF_model.py           # Fase A: addestra un modello Random Forest e lo salva
│   ├── explain.py            # Fase B: genera le spiegazioni XAI dal modello
│   └── stimoli_xai.py        # funzioni di supporto: mappe italiane, generazione stimoli
│                              # SHAP/DiCE/Anchors; importato dal notebook 02, non eseguito da solo
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
git clone https://github.com/elia3008/Tesi-XAI-Elia-Corti.git
cd Tesi-XAI-Elia-Corti

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
- Le versioni esatte delle librerie sono in `requirements.txt` — un freeze
  completo dell'ambiente funzionante (non solo le dipendenze dirette), per
  garantire un'installazione riproducibile fino in fondo. La versione di
  riferimento per gli explainer è `alibi==0.5.5`, che non impone vincoli su
  numpy: nessun conflitto con `shap==0.52.0` (che invece richiede numpy>=2).
- La cartella `artifacts/` non è presente (vedi `.gitignore`): va rigenerata
  eseguendo la pipeline nell'ordine indicato sopra.

---

## Dataset

Heart Disease UCI — versione con missing imputation (Kaggle/ZTM), 303 righe.  
Fonte originale: UCI Machine Learning Repository, Cleveland dataset (id=45).  
Target: 0 = assenza di malattia cardiaca, 1 = presenza.