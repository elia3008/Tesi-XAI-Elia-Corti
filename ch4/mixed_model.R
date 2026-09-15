# =============================================================================
# Analisi statistica del Capitolo 4 -- PARTE R (modello a effetti misti)
# =============================================================================
# Legge l'output di analisi_python.py (simulatability_items.csv, age_group.csv)
# e stima con lme4::glmer (approssimazione di Laplace) tutti i confronti D1 ed
# eta' presentati come "modello a effetti misti" nel capitolo. Da eseguire
# nella cartella output/ prodotta dallo script Python.
#
# Richiede: install.packages("lme4")  (su Ubuntu: apt-get install r-cran-lme4)
# =============================================================================

library(lme4)

sim <- read.csv("simulatability_items.csv")
age <- read.csv("age_group.csv")
colnames(age) <- c("participant_id", "age_group")
sim <- merge(sim, age, by = "participant_id")

fit_or <- function(df, formula_str, term) {
  m <- glmer(as.formula(formula_str), data = df, family = binomial,
             control = glmerControl(optimizer = "bobyqa"))
  est <- fixef(m)[term]
  se  <- sqrt(diag(vcov(m)))[term]
  p   <- 2 * pnorm(-abs(est / se))
  r <- c(exp(est), exp(est - 1.96 * se), exp(est + 1.96 * se), p)
  names(r) <- c("OR", "CI_low", "CI_high", "p")
  r
}

cat("=== Tecnico vs non tecnico a Baseline ===\n")
s <- sim[sim$block == "Baseline", ]
s$tecnico <- as.integer(s$group == "tecnico")
print(fit_or(s, "correct ~ tecnico + (1|participant_id)", "tecnico"))

cat("\n=== Effetto di ciascun metodo entro ciascun gruppo (Tab 4.3/4.4) ===\n")
p_vals <- c()
for (grp in c("non_tecnico", "tecnico")) {
  for (metodo in c("SHAP", "DiCE", "Anchors")) {
    s <- sim[sim$block %in% c("Baseline", metodo) & sim$group == grp, ]
    s$is_m <- as.integer(s$block == metodo)
    r <- fit_or(s, "correct ~ is_m + (1|participant_id)", "is_m")
    cat(grp, metodo, ":", round(r, 4), "\n")
    p_vals <- c(p_vals, r["p"])
  }
}
cat("p Holm:", round(p.adjust(p_vals, method = "holm"), 4), "\n")

cat("\n=== Interazione blocco x gruppo (Tab 4.5) ===\n")
p_vals <- c()
for (metodo in c("SHAP", "DiCE", "Anchors")) {
  s <- sim[sim$block %in% c("Baseline", metodo), ]
  s$is_m <- as.integer(s$block == metodo)
  s$tecnico <- as.integer(s$group == "tecnico")
  r <- fit_or(s, "correct ~ is_m * tecnico + (1|participant_id)", "is_m:tecnico")
  cat(metodo, ":", round(r, 4), "\n")
  p_vals <- c(p_vals, r["p"])
}
cat("p Holm:", round(p.adjust(p_vals, method = "holm"), 4), "\n")

cat("\n=== Confronti a coppie pooled (Tab 4.6) ===\n")
pairs <- list(c("Baseline", "SHAP"), c("Baseline", "DiCE"), c("Baseline", "Anchors"),
              c("SHAP", "DiCE"), c("SHAP", "Anchors"), c("DiCE", "Anchors"))
p_vals <- c()
for (pr in pairs) {
  s <- sim[sim$block %in% pr, ]
  s$is_b <- as.integer(s$block == pr[2])
  r <- fit_or(s, "correct ~ is_b + (1|participant_id)", "is_b")
  cat(pr[1], "vs", pr[2], ":", round(r, 4), "\n")
  p_vals <- c(p_vals, r["p"])
}
cat("p Holm:", round(p.adjust(p_vals, method = "holm"), 4), "\n")

cat("\n=== Eta': effetto principale + interazioni ===\n")
for (metodo in c("SHAP", "DiCE", "Anchors")) {
  s <- sim[sim$block %in% c("Baseline", metodo), ]
  s$is_m <- as.integer(s$block == metodo)
  s$is_25p <- as.integer(s$age_group == "25+")
  cat("--", metodo, "--\n")
  print(fit_or(s, "correct ~ is_m * is_25p + (1|participant_id)", "is_m"))
  print(fit_or(s, "correct ~ is_m * is_25p + (1|participant_id)", "is_25p"))
  print(fit_or(s, "correct ~ is_m * is_25p + (1|participant_id)", "is_m:is_25p"))
}

cat("\n=== Componente di varianza (per l'ICC citato nel testo, dal modello tecnico Baseline+SHAP) ===\n")
s <- sim[sim$block %in% c("Baseline", "SHAP") & sim$group == "tecnico", ]
s$is_shap <- as.integer(s$block == "SHAP")
m_icc <- glmer(correct ~ is_shap + (1|participant_id), data = s, family = binomial,
               control = glmerControl(optimizer = "bobyqa"))
print(VarCorr(m_icc))
var_participant <- as.numeric(VarCorr(m_icc)$participant_id[1])
icc <- var_participant / (var_participant + (pi^2 / 3))  # varianza residua logistica standard = pi^2/3
cat("ICC approssimato (formula latente logistica):", round(icc, 4), "\n")