"""
Analisi statistica del Capitolo 4 -- PARTE PYTHON
==============================================================================
Copre: caricamento e pulizia dati, statistiche descrittive, tutti i confronti
D1 ed eta' stimati con GEE, D2 (correlazione/regressione), calibrazione della
confidenza (modello lineare misto), asimmetria di SHAP, affidabilita' e
confronti a coppie sulla scala ESS.

I confronti D1 ed eta' stimati con il modello a effetti misti (lme4) sono
nello script R separato (mixed_model.R), che legge l'output di questo
script (simulatability_items.csv) come input. Servono entrambi per
riprodurre per intero il Capitolo 4: nessun pacchetto Python disponibile
stima in modo affidabile un modello a effetti misti logistico per questo
tipo di dati (si veda la nota in apertura del capitolo).

Input : Dati_questionario_XAI.xlsx (foglio 'risposte', formato lungo)
Output: CSV con tutte le tabelle, in OUT_DIR
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon, friedmanchisquare
import statsmodels.formula.api as smf
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial
from statsmodels.stats.multitest import multipletests

SRC = "Dati_questionario_XAI.xlsx"
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

BLOCK_ORDER = ["Baseline", "SHAP", "DiCE", "Anchors"]
METHOD_ORDER = ["SHAP", "DiCE", "Anchors"]
TROPPO_VELOCE_SOGLIA_SEC = 385  # isola i 3 completamenti nettamente piu' rapidi
                                 # del resto del campione (il successivo e' a 6.47 min)


# =============================================================================
# 1. CARICAMENTO E PULIZIA (Sezione 4.1 - Campione)
# =============================================================================
COLS = ["participant_id", "group", "block_sequence", "block_position", "block",
        "category", "item", "response", "correct", "confidence", "duration", "timestamp"]

df = pd.read_excel(SRC, sheet_name="risposte", header=None, names=COLS)
assert (df.groupby("participant_id").size() == 53).all(), "atteso 53 righe/partecipante"

attention = (df[df.category == "attention"]
             .set_index("participant_id")["correct"].rename("attention_pass").astype(int))
durata_totale = (df[(df.category == "timing") & (df.item == "total")]
                  .set_index("participant_id")["duration"])
troppo_veloce = durata_totale[durata_totale < TROPPO_VELOCE_SOGLIA_SEC].index.tolist()
esclusi = list(set(attention[attention == 0].index) | set(troppo_veloce))

df["valido"] = ~df.participant_id.isin(esclusi)
df_v = df[df.valido].copy()

participants = (df_v.groupby("participant_id")
                .agg(group=("group", "first"), order=("block_sequence", "first")).reset_index())
participants.to_csv(f"{OUT_DIR}/participants.csv", index=False)
print(f"N totale: {df.participant_id.nunique()}  |  Esclusi: {len(esclusi)}  |  N valido: {len(participants)}")
print(participants.group.value_counts())


# =============================================================================
# 2. TABELLE DI BASE (simulatability item-level, accuratezza, ESS)
# =============================================================================
sim = df_v[df_v.category == "simulatability"][
    ["participant_id", "group", "block_position", "block", "item", "correct", "confidence"]
].copy()
sim["correct"] = sim["correct"].astype(int)
sim["block"] = pd.Categorical(sim["block"], categories=BLOCK_ORDER, ordered=True)
sim.to_csv(f"{OUT_DIR}/simulatability_items.csv", index=False)  # <-- input per lo script R
print(f"\nN item simulatability (atteso {len(participants) * 20}): {len(sim)}")

block_acc = (sim.groupby(["participant_id", "group", "block"], observed=True)
             .agg(accuracy=("correct", "mean")).reset_index())
block_acc.to_csv(f"{OUT_DIR}/block_accuracy.csv", index=False)

ess_raw = df_v[df_v.category == "satisfaction"][["participant_id", "group", "block", "item", "response"]].copy()
ess_raw["response"] = ess_raw["response"].astype(float)
ess_by_block = ess_raw.groupby(["participant_id", "block"]).agg(ess_mean=("response", "mean")).reset_index()
ess_by_block.to_csv(f"{OUT_DIR}/ess_by_block.csv", index=False)

demo = df_v[df_v.category == "demographics"].pivot(index="participant_id", columns="item", values="response")
age_group = demo["age"].map(lambda a: "18-24" if a == "18-24" else "25+")
age_group.to_csv(f"{OUT_DIR}/age_group.csv")  # <-- input per lo script R
sim = sim.merge(age_group.rename("age_group"), left_on="participant_id", right_index=True)


# =============================================================================
# 3. TABELLE DESCRITTIVE (Tabella 4.1, griglia blocco x gruppo)
# =============================================================================
print("\n=== Tabella 4.1: accuratezza per blocco ===")
desc = block_acc.groupby("block", observed=True)["accuracy"].agg(["mean", "std", "count"])
desc["se"] = desc["std"] / np.sqrt(desc["count"])
desc["ci95_low"] = desc["mean"] - 1.96 * desc["se"]
desc["ci95_high"] = desc["mean"] + 1.96 * desc["se"]
print(desc.round(3))
desc.round(4).to_csv(f"{OUT_DIR}/tab_4_1_descrittive.csv")

print("\n=== Griglia blocco x gruppo ===")
grid = block_acc.groupby(["block", "group"], observed=True)["accuracy"].mean().unstack()
print(grid.round(3))
grid.round(4).to_csv(f"{OUT_DIR}/tab_grid.csv")


# =============================================================================
# 4. D1 -- TUTTI I CONFRONTI CON GEE
# =============================================================================
def gee_or(data, formula):
    """Adatta un GEE binomiale (correlazione exchangeable, errori robusti,
    cluster = participant_id) e restituisce OR, IC 95% e p-value per l'ultimo
    termine della formula."""
    m = GEE.from_formula(formula, groups="participant_id", data=data,
                          family=Binomial(), cov_struct=Exchangeable()).fit()
    term = m.params.index[-1]
    est, se = m.params[term], m.bse[term]
    return {"or": np.exp(est), "ci_low": np.exp(est - 1.96 * se),
            "ci_high": np.exp(est + 1.96 * se), "p_value": m.pvalues[term]}


print("\n=== GEE: tecnico vs non tecnico a Baseline ===")
s = sim[sim.block == "Baseline"].copy()
s["tecnico"] = (s.group == "tecnico").astype(int)
r = gee_or(s, "correct ~ tecnico")
print(r)
pd.DataFrame([r]).to_csv(f"{OUT_DIR}/gee_gruppo_baseline.csv", index=False)

print("\n=== GEE: effetto di ciascun metodo entro ciascun gruppo (Tab 4.3/4.4) ===")
rows = []
for grp in ["non_tecnico", "tecnico"]:
    for metodo in METHOD_ORDER:
        s = sim[(sim.block.isin(["Baseline", metodo])) & (sim.group == grp)].copy()
        s["is_m"] = (s.block == metodo).astype(int)
        r = gee_or(s, "correct ~ is_m")
        r.update({"gruppo": grp, "metodo": metodo})
        rows.append(r)
tab_grp = pd.DataFrame(rows)
_, p_holm, _, _ = multipletests(tab_grp["p_value"], method="holm")
tab_grp["p_holm"] = p_holm
print(tab_grp.round(4).to_string(index=False))
tab_grp.to_csv(f"{OUT_DIR}/gee_tab_4_3_4_4.csv", index=False)

print("\n=== GEE: interazione blocco x gruppo (Tab 4.5) ===")
rows = []
for metodo in METHOD_ORDER:
    s = sim[sim.block.isin(["Baseline", metodo])].copy()
    s["is_m"] = (s.block == metodo).astype(int)
    s["tecnico"] = (s.group == "tecnico").astype(int)
    r = gee_or(s, "correct ~ is_m * tecnico")
    r["metodo"] = metodo
    rows.append(r)
tab_int = pd.DataFrame(rows)
_, p_holm, _, _ = multipletests(tab_int["p_value"], method="holm")
tab_int["p_holm"] = p_holm
print(tab_int.round(4).to_string(index=False))
tab_int.to_csv(f"{OUT_DIR}/gee_tab_4_5_interazione.csv", index=False)

print("\n=== GEE: confronti a coppie pooled (Tab 4.6) ===")
pairs = [("Baseline", "SHAP"), ("Baseline", "DiCE"), ("Baseline", "Anchors"),
         ("SHAP", "DiCE"), ("SHAP", "Anchors"), ("DiCE", "Anchors")]
rows = []
for a, b in pairs:
    s = sim[sim.block.isin([a, b])].copy()
    s["is_b"] = (s.block == b).astype(int)
    r = gee_or(s, "correct ~ is_b")
    r["confronto"] = f"{a} vs {b}"
    rows.append(r)
tab_pw = pd.DataFrame(rows)
_, p_holm, _, _ = multipletests(tab_pw["p_value"], method="holm")
tab_pw["p_holm"] = p_holm
print(tab_pw.round(4).to_string(index=False))
tab_pw.to_csv(f"{OUT_DIR}/gee_tab_4_6_pairwise.csv", index=False)

print("\n=== Controllo di robustezza: Friedman su medie per partecipante ===")
wide = block_acc.pivot(index="participant_id", columns="block", values="accuracy")[BLOCK_ORDER]
stat, p = friedmanchisquare(*[wide[b] for b in BLOCK_ORDER])
print(f"chi2={stat:.3f}  p={p:.5f}")

print("\n=== GEE: eta' (effetto principale + interazioni) ===")
rows = []
for metodo in METHOD_ORDER:
    s = sim[sim.block.isin(["Baseline", metodo])].copy()
    s["is_m"] = (s.block == metodo).astype(int)
    s["is_25p"] = (s.age_group == "25+").astype(int)
    m = GEE.from_formula("correct ~ is_m * is_25p", groups="participant_id", data=s,
                          family=Binomial(), cov_struct=Exchangeable()).fit()
    for term in ["is_m", "is_25p", "is_m:is_25p"]:
        est, se = m.params[term], m.bse[term]
        rows.append({"metodo": metodo, "termine": term, "or": np.exp(est),
                     "ci_low": np.exp(est - 1.96 * se), "ci_high": np.exp(est + 1.96 * se),
                     "p_value": m.pvalues[term]})
tab_age = pd.DataFrame(rows)
print(tab_age.round(4).to_string(index=False))
tab_age.to_csv(f"{OUT_DIR}/gee_eta.csv", index=False)


# =============================================================================
# 5. UN PATTERN NELL'ERRORE: ASIMMETRIA SHAP PER CLASSE REALE
# =============================================================================
print("\n=== Accuratezza SHAP per classe reale del paziente ===")
# verdetti presi dalla Tabella A.4 (Appendice A), verificati sul repository
verdetti = {"B_T1": "M", "B_T2": "M", "B_T3": "M", "B_T4": "S", "B_T5": "S",
            "S_T1": "S", "S_T2": "M", "S_T3": "M", "S_T4": "S", "S_T5": "M",
            "D_T1": "S", "D_T2": "M", "D_T3": "M", "D_T4": "S", "D_T5": "M",
            "A_T1": "S", "A_T2": "M", "A_T3": "S", "A_T4": "M", "A_T5": "S"}
sim["verdetto"] = sim["item"].map(verdetti)
shap_asym = sim[sim.block == "SHAP"].groupby("verdetto")["correct"].agg(["mean", "count"])
print(shap_asym)
shap_asym.to_csv(f"{OUT_DIR}/shap_asimmetria.csv")


# =============================================================================
# 6. D2 -- CORRELAZIONE E PENDENZE (modello lineare misto, MixedLM)
# =============================================================================
rq2 = block_acc.merge(ess_by_block, on=["participant_id", "block"], how="left")
rq2_expl = rq2[rq2.block != "Baseline"].dropna(subset=["ess_mean"]).copy()
rq2_expl["block"] = pd.Categorical(rq2_expl["block"], categories=METHOD_ORDER, ordered=True)

print("\n=== Correlazioni di Spearman ESS~accuratezza ===")
righe = [{"subset": "Tutti", "n": len(rq2_expl),
          **dict(zip(["rho", "p"], spearmanr(rq2_expl.ess_mean, rq2_expl.accuracy)))}]
for m in METHOD_ORDER:
    sub = rq2_expl[rq2_expl.block == m]
    righe.append({"subset": m, "n": len(sub), **dict(zip(["rho", "p"], spearmanr(sub.ess_mean, sub.accuracy)))})
corr_tab = pd.DataFrame(righe)
print(corr_tab.round(4).to_string(index=False))
corr_tab.to_csv(f"{OUT_DIR}/rq2_correlazioni.csv", index=False)

res_diretto = smf.mixedlm("ess_mean ~ accuracy + C(block, Treatment('SHAP')) + C(group)",
                           data=rq2_expl, groups=rq2_expl["participant_id"]).fit(reml=True)
print(f"\nEffetto diretto accuratezza su ESS: b={res_diretto.params['accuracy']:.3f}  "
      f"p={res_diretto.pvalues['accuracy']:.4f}")

res_int = smf.mixedlm("ess_mean ~ accuracy * C(block, Treatment('SHAP')) + C(group)",
                       data=rq2_expl, groups=rq2_expl["participant_id"]).fit(reml=True)
params, cov = res_int.params, res_int.cov_params()
combinazioni = {"SHAP": ["accuracy"],
                "DiCE": ["accuracy", "accuracy:C(block, Treatment('SHAP'))[T.DiCE]"],
                "Anchors": ["accuracy", "accuracy:C(block, Treatment('SHAP'))[T.Anchors]"]}
print("\n=== Pendenze ESS~accuratezza per metodo ===")
from scipy.stats import norm
righe_pend = []
for metodo, termini in combinazioni.items():
    stima = sum(params[t] for t in termini)
    se = np.sqrt(sum(cov.loc[i, j] for i in termini for j in termini))
    p = 2 * (1 - norm.cdf(abs(stima / se)))
    righe_pend.append({"metodo": metodo, "pendenza": stima, "ci_low": stima - 1.96 * se,
                        "ci_high": stima + 1.96 * se, "p": p})
    print(f"  {metodo:10s} pendenza={stima:.3f}  p={p:.4f}")
pd.DataFrame(righe_pend).to_csv(f"{OUT_DIR}/rq2_pendenze.csv", index=False)


# =============================================================================
# 7. AFFIDABILITA' E CONFRONTI A COPPIE SULLA SCALA ESS
# =============================================================================
ess_items = df_v[df_v.category == "satisfaction"][["participant_id", "block", "item", "response"]].copy()
ess_items["response"] = ess_items["response"].astype(float)

def cronbach_alpha(wide):
    k = wide.shape[1]
    item_vars = wide.var(axis=0, ddof=1)
    total_var = wide.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - item_vars.sum() / total_var)

print("\n=== Alpha di Cronbach per blocco ===")
for metodo in METHOD_ORDER:
    wide = ess_items[ess_items.block == metodo].pivot(index="participant_id", columns="item", values="response")
    print(f"{metodo}: alpha={cronbach_alpha(wide.dropna()):.3f}")

print("\n=== Friedman e confronti a coppie sui punteggi ESS ===")
wide_ess = ess_by_block.pivot(index="participant_id", columns="block", values="ess_mean")[METHOD_ORDER]
stat, p = friedmanchisquare(wide_ess["SHAP"], wide_ess["DiCE"], wide_ess["Anchors"])
print(f"Friedman: chi2={stat:.3f}  p={p:.5f}")
for a, b in [("SHAP", "DiCE"), ("SHAP", "Anchors"), ("DiCE", "Anchors")]:
    stat, p = wilcoxon(wide_ess[a], wide_ess[b])
    print(f"  {a} vs {b}: p={p:.4f}  (medie: {wide_ess[a].mean():.2f} vs {wide_ess[b].mean():.2f})")


# =============================================================================
# 8. CALIBRAZIONE DELLA CONFIDENZA (MixedLM) -- generale e per eta'
# =============================================================================
sim["correct_f"] = sim["correct"].map({0: "errato", 1: "corretto"})

print("\n=== Gap di calibrazione per blocco ===")
gap = sim.groupby(["block", "correct"], observed=True)["confidence"].mean().unstack()
gap.columns = ["conf_errato", "conf_corretto"]
gap["gap"] = gap["conf_corretto"] - gap["conf_errato"]
print(gap.round(3))
gap.to_csv(f"{OUT_DIR}/calibrazione_gap.csv")

formula_cal = "confidence ~ C(correct_f, Treatment('errato')) * C(block, Treatment('Baseline')) + C(group)"
res_cal = smf.mixedlm(formula_cal, data=sim, groups=sim["participant_id"]).fit(reml=True)
print("\n=== Modello di calibrazione ===")
for metodo in METHOD_ORDER:
    term = f"C(correct_f, Treatment('errato'))[T.corretto]:C(block, Treatment('Baseline'))[T.{metodo}]"
    print(f"  {metodo}: beta={res_cal.params[term]:.3f}  p={res_cal.pvalues[term]:.4f}")

formula_cal_age = ("confidence ~ C(correct_f, Treatment('errato')) * C(block, Treatment('Baseline')) "
                    "* C(age_group, Treatment('18-24'))")
res_cal_age = smf.mixedlm(formula_cal_age, data=sim, groups=sim["participant_id"]).fit(reml=True)
print("\n=== Calibrazione per eta' (modello a tre vie) ===")
for metodo in METHOD_ORDER:
    term = (f"C(correct_f, Treatment('errato'))[T.corretto]:C(block, Treatment('Baseline'))[T.{metodo}]:"
            f"C(age_group, Treatment('18-24'))[T.25+]")
    print(f"  {metodo} x 25+: beta={res_cal_age.params[term]:.3f}  p={res_cal_age.pvalues[term]:.4f}")

print(f"\nScript completato. Output in: {os.path.abspath(OUT_DIR)}")
print("Prossimo passo: esegui analisi_R_lme4.R nella stessa cartella per i")
print("confronti D1 ed eta' stimati con il modello a effetti misti.")


# =============================================================================
# 9. ICC -- NOTA SULLE DUE CONVENZIONI (approssimazione lineare vs formula
#    a variabile latente logistica; lo script R calcola quest'ultima)
# =============================================================================
print("\n=== ICC, approssimazione lineare (quella citata nel testo, 0.018) ===")
s = sim[sim.block.isin(["Baseline", "SHAP"]) & (sim.group == "tecnico")].copy()
m_icc = smf.mixedlm("correct ~ 1", data=s, groups=s["participant_id"]).fit(reml=True)
var_participant = m_icc.cov_re.iloc[0, 0]
var_resid = m_icc.scale
icc_lineare = var_participant / (var_participant + var_resid)
print(f"Varianza tra partecipanti: {var_participant:.4f}  |  Varianza residua: {var_resid:.4f}")
print(f"ICC (approssimazione lineare): {icc_lineare:.4f}")
print("Nota: lo script R calcola anche l'ICC con la formula a variabile latente")
print("logistica (sigma^2_u / (sigma^2_u + pi^2/3)) sullo stesso sottoinsieme di")
print("dati: da' un valore numericamente piu' piccolo (~0.002), convenzione")
print("diversa per lo stesso fenomeno -- entrambe indicano una variabilita' tra")
print("partecipanti molto bassa, che e' la conclusione sostanziale riportata nel testo.")