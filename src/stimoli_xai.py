"""Generazione degli stimoli DiCE e Anchors per il questionario.

Da salvare in src/stimoli_xai.py e importare dal notebook 02.

Contiene l'unica copia autorevole di label_map / value_maps: importale da qui
anche nelle celle SHAP, così le tre spiegazioni non possono divergere.

Codifica assunta (Cleveland grezzo): cp 1-4, slope 1=ascendente/3=discendente,
thal 3/6/7. Se cambi CSV, cambia SOLO value_maps qui sotto.
"""
from pathlib import Path
import math
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --------------------------------------------------------------------------
# 1. Mappe italiane (fonte unica per SHAP, DiCE, Anchors e tabelle-paziente)
# --------------------------------------------------------------------------

label_map = {
    'age': 'Età', 'sex': 'Sesso', 'cp': 'Tipo dolore toracico',
    'trestbps': 'Pressione a riposo', 'chol': 'Colesterolo', 'fbs': 'Glicemia a digiuno',
    'restecg': 'ECG a riposo', 'thalach': 'Freq. cardiaca max', 'exang': 'Angina da sforzo',
    'oldpeak': 'Depressione tratto ST', 'slope': 'Pendenza tratto ST',
    'ca': 'Vasi colorati', 'thal': 'Test al tallio',
}

value_maps = {
    'sex':     {0: 'Femmina', 1: 'Maschio'},
    'cp':      {1: 'Angina tipica', 2: 'Angina atipica', 3: 'Dolore non anginoso', 4: 'Asintomatico'},
    # ATTENZIONE: nessuna etichetta deve contenere '<' o '>': alibi in quel caso
    # omette il nome della feature dalla regola (vedi costruisci_categorical_names)
    'fbs':     {0: 'Normale (fino a 120 mg/dL)', 1: 'Alta (oltre 120 mg/dL)'},
    'restecg': {0: 'Normale', 1: 'Anomalia ST-T', 2: 'Ipertrofia ventric.'},
    'exang':   {0: 'No', 1: 'Sì'},
    'slope':   {1: 'Ascendente', 2: 'Piatta', 3: 'Discendente'},
    'thal':    {3: 'Normale', 6: 'Difetto fisso', 7: 'Difetto reversibile'},
}

unita = {
    'age': 'anni', 'trestbps': 'mmHg', 'chol': 'mg/dl',
    'thalach': 'bpm', 'oldpeak': 'mm', 'ca': 'vasi',
}

# Palette identica a tabella_paziente: lo stile non deve confondere il confronto
C_HEADER = '#1F4E79'
C_RIGA_A = '#FFFFFF'
C_RIGA_B = '#F2F5F9'
C_BORDO  = '#B7C6D9'
C_VALORE = '#1F4E79'
C_TESTO  = '#1A1A1A'
C_CAMBIO = '#C0392B'   # valore modificato nei controfattuali
C_GRIGIO = '#8A8A8A'


# --------------------------------------------------------------------------
# 2. Formattazione valori (NaN-safe)
# --------------------------------------------------------------------------

def manca(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def fmt_valore(v):
    if manca(v):
        return "n.d."
    v = float(v)
    return str(int(v)) if v.is_integer() else f"{v:g}"


def descrivi_valore(col, v):
    """Valore leggibile: etichetta per le categoriche, numero + unità per le continue."""
    if manca(v):
        return "Dato mancante"
    if col in value_maps:
        return value_maps[col].get(int(v), f"(codice {fmt_valore(v)})")
    u = unita.get(col)
    return f"{fmt_valore(v)} {u}" if u else fmt_valore(v)


def etichetta(col):
    return label_map.get(col, col)


def pazienti_completi(X):
    """Indici posizionali dei pazienti senza valori mancanti: solo questi vanno
    usati come stimoli, altrimenti DiCE e Anchors ricevono NaN."""
    return np.where(X.notna().all(axis=1))[0]


# --------------------------------------------------------------------------
# 3. ANCHORS
# --------------------------------------------------------------------------

def costruisci_categorical_names(colonne):
    """alibi indicizza le etichette per posizione: categorical_names[j][int(valore)].

    Con la codifica Cleveland grezza i codici non partono da 0 (thal = 3/6/7),
    quindi si costruisce una lista lunga max(codice)+1 con dei segnaposto nelle
    posizioni inutilizzate. I segnaposto non compaiono mai nelle regole perché
    quei valori non esistono nei dati.
    """
    cat_names = {}
    for j, c in enumerate(colonne):
        if c in value_maps:
            for etich in value_maps[c].values():
                if '<' in etich or '>' in etich:
                    raise ValueError(
                        f"L'etichetta '{etich}' di '{c}' contiene < o >: alibi in quel caso "
                        f"omette il nome della feature dalla regola. Riscrivila a parole "
                        f"(es. 'fino a 120 mg/dL', 'oltre 120 mg/dL').")
            vmax = max(value_maps[c])
            cat_names[j] = [value_maps[c].get(k, f"(codice {k})") for k in range(vmax + 1)]
    return cat_names


def crea_spiegatore_anchors(model, X_train, disc_perc=(25, 50, 75), seed=123):
    """AnchorTabular con le categoriche dichiarate: le regole diventano
    'Test al tallio = Difetto reversibile' invece di 'thal > 3.00'."""
    from alibi.explainers import AnchorTabular

    colonne = list(X_train.columns)

    def predict_fn(x):
        # il modello è stato addestrato su DataFrame: reincapsulo per evitare
        # il warning sui feature names e garantire l'ordine delle colonne
        return model.predict(pd.DataFrame(x, columns=colonne))

    expl = AnchorTabular(
        predict_fn,
        feature_names=colonne,
        categorical_names=costruisci_categorical_names(colonne),
        seed=seed,
    )
    expl.fit(X_train.values, disc_perc=disc_perc)
    return expl, predict_fn


_RE_NUM = re.compile(r"(-?\d+\.\d+)")


def traduci_predicato(p, colonne):
    """'thalach <= 105.00' -> 'Freq. cardiaca max <= 105'."""
    for c in sorted(colonne, key=len, reverse=True):
        p = re.sub(rf"\b{re.escape(c)}\b", etichetta(c), p)
    return _RE_NUM.sub(lambda m: fmt_valore(float(m.group(1))), p)


def figura_anchor(spiegazione, colonne, classe, mostra_metriche=True,
                  salva=None, dpi=200, larghezza=8.2):
    """Card 'SE ... E ... ALLORA' con lo stesso stile delle tabelle-paziente."""
    predicati = [traduci_predicato(p, colonne) for p in spiegazione.anchor]
    if not predicati:
        predicati = ["(nessuna condizione trovata: la previsione non dipende "
                     "da poche variabili)"]

    esito = "MALATTIA PRESENTE" if int(classe) == 1 else "MALATTIA ASSENTE"
    n = len(predicati)
    h_riga = 0.46
    h_tot = h_riga * (n + 3.4) + (0.5 if mostra_metriche else 0)

    fig, ax = plt.subplots(figsize=(larghezza, h_tot))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, h_tot)
    ax.axis('off')

    y = h_tot - h_riga
    ax.add_patch(Rectangle((0, y), 1, h_riga, facecolor=C_HEADER, edgecolor='none'))
    ax.text(0.012, y + h_riga / 2, "Regola usata dal modello per questo paziente",
            ha='left', va='center', color='white', fontsize=10.5, fontweight='bold')

    y -= h_riga * 0.9
    ax.text(0.012, y, "SE", ha='left', va='center', fontsize=10.5,
            fontweight='bold', color=C_HEADER)

    for i, p in enumerate(predicati):
        y -= h_riga
        ax.add_patch(Rectangle((0.06, y - h_riga * 0.38), 0.94, h_riga * 0.86,
                               facecolor=C_RIGA_A if i % 2 == 0 else C_RIGA_B,
                               edgecolor=C_BORDO, linewidth=0.8))
        ax.text(0.075, y, p, ha='left', va='center', fontsize=10, color=C_TESTO)
        if i < len(predicati) - 1:
            ax.text(0.032, y - h_riga * 0.5, "E", ha='center', va='center',
                    fontsize=9.5, fontweight='bold', color=C_HEADER)

    y -= h_riga * 1.15
    ax.text(0.012, y, "ALLORA il modello prevede:", ha='left', va='center',
            fontsize=10.5, fontweight='bold', color=C_HEADER)
    ax.text(0.42, y, esito, ha='left', va='center', fontsize=10.5,
            fontweight='bold', color=C_CAMBIO if int(classe) == 1 else C_VALORE)

    if mostra_metriche:
        y -= h_riga * 0.85
        ax.text(0.012, y,
                f"La regola è corretta nel {spiegazione.precision * 100:.0f}% dei pazienti "
                f"simili e si applica al {spiegazione.coverage * 100:.0f}% dei pazienti.",
                ha='left', va='center', fontsize=8.5, color=C_GRIGIO, style='italic')

    plt.tight_layout()
    if salva:
        Path(salva).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(salva, dpi=dpi, bbox_inches='tight', facecolor='white')
    return fig


# --------------------------------------------------------------------------
# 4. DiCE
# --------------------------------------------------------------------------

def crea_spiegatore_dice(model, X_train, y_train, cont_features, method="genetic"):
    import dice_ml

    df_train = X_train.copy()
    for c in df_train.columns:
        df_train[c] = df_train[c].astype("float64")
    df_train["target"] = np.asarray(y_train, dtype="float64")

    d = dice_ml.Data(dataframe=df_train,
                     continuous_features=list(cont_features),
                     outcome_name="target")
    m = dice_ml.Model(model=model, backend="sklearn")
    return dice_ml.Dice(d, m, method=method)


def _cf_dataframe(cf_result):
    """final_cfs_df_sparse quando disponibile (meno feature modificate)."""
    ex = cf_result.cf_examples_list[0]
    df = ex.final_cfs_df_sparse if ex.final_cfs_df_sparse is not None else ex.final_cfs_df
    return df.drop(columns=["target"], errors="ignore").reset_index(drop=True)


def _testo_adattato(fig, ax, x, y, larghezza_cella, testo, fontsize,
                    margine=0.92, minimo=6.0, **kw):
    """Disegna il testo riducendo il corpo finché non entra nella cella.

    Serve perché il testo bianco delle intestazioni, se deborda dal rettangolo
    colorato, finisce su fondo bianco e sembra tagliato.
    """
    t = ax.text(x, y, testo, fontsize=fontsize, **kw)
    try:
        ren = fig.canvas.get_renderer()
        x0, x1 = ax.transAxes.transform((0, 0))[0], ax.transAxes.transform((1, 0))[0]
        disponibile = (x1 - x0) * larghezza_cella * margine
        while fontsize > minimo and t.get_window_extent(renderer=ren).width > disponibile:
            fontsize -= 0.5
            t.set_fontsize(fontsize)
    except Exception:
        pass   # backend senza renderer: si tiene il corpo originale
    return t


def figura_controfattuali(originale, cf_df, classe_originale, salva=None, dpi=200,
                          altezza_riga=0.44, larghezze=(0.34, 0.28, 0.38)):
    """Tabella 'Valore attuale -> Valore da raggiungere', solo le variabili che cambiano.

    originale: pd.Series del paziente (indicizzata sui nomi colonna originali)
    cf_df:     DataFrame dei controfattuali (una riga per scenario)
    """
    cambiate = [c for c in cf_df.columns
                if any(not np.isclose(float(cf_df.loc[i, c]), float(originale[c]))
                       for i in cf_df.index)]
    if not cambiate:
        raise ValueError("Nessuna variabile modificata: controlla features_to_vary.")

    n_cf = len(cf_df)
    larghezze = np.array(larghezze[:2] + (larghezze[2],) * n_cf, dtype=float)
    larghezze = larghezze / larghezze.sum()
    x = np.concatenate([[0.0], np.cumsum(larghezze)])

    n = len(cambiate)
    h_tot = altezza_riga * (n + 2.6)
    fig, ax = plt.subplots(figsize=(5.2 + 2.7 * n_cf, h_tot))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n + 2.6)
    ax.axis('off')

    intestazioni = ['Variabile', 'Valori reali'] + \
                   ([f'Scenario {i + 1}' for i in range(n_cf)] if n_cf > 1 else ['Valori ipotetici'])
    for j, testo in enumerate(intestazioni):
        ax.add_patch(Rectangle((x[j], n), larghezze[j], 1,
                               facecolor=C_HEADER, edgecolor='white', linewidth=1.2))
        _testo_adattato(fig, ax, x[j] + larghezze[j] / 2, n + 0.5, larghezze[j], testo, 10,
                        ha='center', va='center', color='white', fontweight='bold')

    for i, col in enumerate(cambiate):
        y = n - 1 - i
        sfondo = C_RIGA_A if i % 2 == 0 else C_RIGA_B
        for j in range(len(intestazioni)):
            ax.add_patch(Rectangle((x[j], y), larghezze[j], 1,
                                   facecolor=sfondo, edgecolor=C_BORDO, linewidth=0.8))
        _testo_adattato(fig, ax, x[0] + 0.010, y + 0.5, larghezze[0] - 0.02,
                        etichetta(col), 9.5, ha='left', va='center',
                        fontweight='bold', color=C_TESTO)
        _testo_adattato(fig, ax, x[1] + larghezze[1] / 2, y + 0.5, larghezze[1],
                        descrivi_valore(col, originale[col]), 9.5,
                        ha='center', va='center', color=C_TESTO)
        for k, idx_cf in enumerate(cf_df.index):
            v = cf_df.loc[idx_cf, col]
            uguale = np.isclose(float(v), float(originale[col]))
            _testo_adattato(fig, ax, x[2 + k] + larghezze[2 + k] / 2, y + 0.5,
                            larghezze[2 + k], "—" if uguale else descrivi_valore(col, v), 9.5,
                            ha='center', va='center',
                            fontweight='normal' if uguale else 'bold',
                            color=C_GRIGIO if uguale else C_CAMBIO)

    da = "malattia presente" if int(classe_originale) == 1 else "malattia assente"
    a = "malattia assente" if int(classe_originale) == 1 else "malattia presente"
    ax.text(0.0, n - len(cambiate) - 0.55,
            f"Con queste modifiche la previsione del modello passa da «{da}» a «{a}».",
            ha='left', va='center', fontsize=9, color=C_HEADER, style='italic')

    plt.tight_layout()
    if salva:
        Path(salva).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(salva, dpi=dpi, bbox_inches='tight', facecolor='white')
    return fig


# --------------------------------------------------------------------------
# 5. Generazione in blocco
# --------------------------------------------------------------------------

def _n_modifiche(riga_cf, originale, colonne):
    return int(sum(not np.isclose(float(riga_cf[c]), float(originale[c])) for c in colonne))


def controfattuali_sparsi(dice_exp, originale, variabili, max_modifiche=5,
                          n_scenari=2, pool_CFs=20, posthoc_sparsity_param=0.1):
    """Genera un bacino ampio di controfattuali e tiene solo i più sparsi.

    DiCE non espone un tetto al numero di feature modificate (sparsity_weight e
    posthoc_sparsity_param sono penalità morbide), quindi il vincolo si impone a
    valle: si generano pool_CFs candidati e si tengono gli n_scenari che
    modificano al massimo max_modifiche variabili.

    Restituisce (cf_df, n_modifiche_per_scenario). cf_df vuoto = nessun
    controfattuale sotto la soglia per questo paziente.
    """
    q = originale.to_frame().T.astype("float64")
    cf = dice_exp.generate_counterfactuals(
        q, total_CFs=pool_CFs, desired_class="opposite",
        features_to_vary=variabili, verbose=False,
        posthoc_sparsity_param=posthoc_sparsity_param)
    df = _cf_dataframe(cf)

    conta = df.apply(lambda r: _n_modifiche(r, originale, df.columns), axis=1)
    ordine = conta.sort_values(kind="stable").index
    df, conta = df.loc[ordine], conta.loc[ordine]

    tenuti = conta[conta <= max_modifiche].index[:n_scenari]
    return df.loc[tenuti].reset_index(drop=True), list(conta.loc[tenuti])


def arrotonda_controfattuali(cf_df, X_train, predict_fn, classe_originale):
    """Riporta i valori alla precisione dei dati reali e riverifica la previsione.

    DiCE tratta le continue come reali e produce valori inesistenti nel dataset
    ('43.4 anni', '114.6 mmHg'). Arrotondare li rende leggibili, ma può riportare
    il punto dalla parte sbagliata del confine decisionale: quindi si ricontrolla
    che il controfattuale arrotondato ribalti ancora la previsione, e si scartano
    quelli che non lo fanno più.

    Restituisce (cf_df_arrotondato, n_scartati).
    """
    decimali = {}
    for c in cf_df.columns:
        v = X_train[c].dropna().astype(float)
        decimali[c] = 0 if np.allclose(v % 1, 0) else 1

    arr = cf_df.copy()
    for c in arr.columns:
        arr[c] = arr[c].astype(float).round(decimali[c])

    tiene = np.asarray(predict_fn(arr.values)) != int(classe_originale)
    return arr[tiene].reset_index(drop=True), int((~tiene).sum())


def verifica_coverage(spiegazione, X_train):
    """Frazione del training set che soddisfa davvero la regola.

    Da confrontare con spiegazione.coverage prima di riportare il numero in tesi.
    """
    inv = {c: {v: k for k, v in m.items()} for c, m in value_maps.items()}

    def soddisfa(p):
        m = re.match(r"^([\d.]+) < (\w+) <= ([\d.]+)$", p)
        if m:
            s = X_train[m.group(2)]
            return (s > float(m.group(1))) & (s <= float(m.group(3)))
        m = re.match(r"^(\w+) (<=|>=|<|>) ([\d.]+)$", p)
        if m:
            s, t = X_train[m.group(1)], float(m.group(3))
            return {"<=": s <= t, ">=": s >= t, "<": s < t, ">": s > t}[m.group(2)]
        m = re.match(r"^(\w+) = (.+)$", p)
        if m:
            return X_train[m.group(1)] == inv[m.group(1)][m.group(2)]
        raise ValueError(f"predicato non riconosciuto: {p!r}")

    mask = np.ones(len(X_train), dtype=bool)
    for p in spiegazione.anchor:
        mask &= soddisfa(p).values
    return float(mask.mean())


def genera_stimoli(indici, X_test, model, anchor_exp, predict_fn, dice_exp,
                   modifiable, cartella, total_CFs=2, threshold=0.95,
                   mostra_metriche=True, salva=True, posthoc_sparsity_param=0.1,
                   verbose=True, max_modifiche=None, pool_CFs=20, X_train=None):
    """Genera anchors_paziente_<idx>.png e dice_paziente_<idx>.png per ogni indice.

    modifiable: lista di colonne oppure la stringa "all" per lasciarle tutte libere.
    """
    cartella = Path(cartella)
    colonne = list(X_test.columns)
    variabili = "all" if modifiable == "all" else list(modifiable)
    esiti = []

    for idx in indici:
        riga = X_test.iloc[idx]
        if riga.isna().any():
            if verbose:
                print(f"idx {idx:3d} | SALTATO: valori mancanti")
            continue
        classe = int(predict_fn(X_test.iloc[[idx]].values)[0])

        a = anchor_exp.explain(riga.values, threshold=threshold)
        fig = figura_anchor(a, colonne, classe, mostra_metriche=mostra_metriche,
                            salva=cartella / f"anchors_paziente_{idx}.png" if salva else None)
        if not salva:
            plt.close(fig)

        q = X_test.iloc[[idx]].astype("float64")
        try:
            if max_modifiche is None:
                cf = dice_exp.generate_counterfactuals(
                    q, total_CFs=total_CFs, desired_class="opposite",
                    features_to_vary=variabili, verbose=False,
                    posthoc_sparsity_param=posthoc_sparsity_param)
                cf_df = _cf_dataframe(cf)
            else:
                cf_df, _ = controfattuali_sparsi(
                    dice_exp, riga, variabili, max_modifiche=max_modifiche,
                    n_scenari=total_CFs, pool_CFs=pool_CFs,
                    posthoc_sparsity_param=posthoc_sparsity_param)
                if cf_df.empty:
                    raise ValueError(
                        f"nessun controfattuale con <= {max_modifiche} modifiche "
                        f"su {pool_CFs} candidati")

            if X_train is not None:
                cf_df, scartati = arrotonda_controfattuali(cf_df, X_train, predict_fn, classe)
                if cf_df.empty:
                    raise ValueError(
                        f"tutti i controfattuali ({scartati}) non ribaltano più "
                        f"la previsione una volta arrotondati")

            fig = figura_controfattuali(riga, cf_df, classe,
                                        salva=cartella / f"dice_paziente_{idx}.png" if salva else None)
            if not salva:
                plt.close(fig)
            cambiate = [c for c in cf_df.columns
                        if any(not np.isclose(float(cf_df.loc[i, c]), float(riga[c]))
                               for i in cf_df.index)]
            n_mod = len(cambiate)
        except Exception as err:
            if verbose:
                print(f"idx {idx:3d} | DiCE fallito: {err}")
            cambiate, n_mod = [], None

        esiti.append({"idx": idx, "classe": classe, "n_condizioni": len(a.anchor),
                      "precision": round(float(a.precision), 2),
                      "coverage": round(float(a.coverage), 2),
                      "feature_cambiate": n_mod,
                      "quali": ", ".join(etichetta(c) for c in cambiate)})
        if verbose:
            print(f"idx {idx:3d} | classe {classe} | anchor {len(a.anchor)} cond. "
                  f"(prec {a.precision:.2f}) | cf: {n_mod} feature modificate")

    return pd.DataFrame(esiti)


def frequenza_feature(riepilogo, colonne):
    """Quante volte ogni feature viene modificata dai controfattuali generati.

    Serve a vedere se DiCE sta davvero toccando age/sex o se il problema è teorico.
    """
    conta = {etichetta(c): 0 for c in colonne}
    for quali in riepilogo["quali"].fillna(""):
        for nome in [q.strip() for q in quali.split(",") if q.strip()]:
            conta[nome] = conta.get(nome, 0) + 1
    n = int(riepilogo["feature_cambiate"].notna().sum())
    return (pd.Series(conta, name=f"modificata (su {n} pazienti)")
            .sort_values(ascending=False)
            .to_frame())