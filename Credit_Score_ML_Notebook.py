# 🚀 Credit Score com Machine Learning (Notebook Completo)

# ==============================
# 1. IMPORTS
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

from xgboost import XGBClassifier

# ==============================
# 2. LOAD DATA
# ==============================
# Substitua pelo seu dataset
# df = pd.read_csv('data/credit_data.csv')

# Dataset simulado (caso não tenha ainda)
np.random.seed(42)
n = 5000

df = pd.DataFrame({
    'idade': np.random.randint(21, 70, n),
    'renda': np.random.normal(5000, 2000, n),
    'tempo_emprego': np.random.randint(0, 20, n),
    'qtd_atrasos': np.random.poisson(2, n),
    'utilizacao_limite': np.random.uniform(0, 1, n),
})

# Target (default)
df['default'] = (
    (df['qtd_atrasos'] > 2).astype(int) +
    (df['utilizacao_limite'] > 0.7).astype(int)
)
df['default'] = (df['default'] > 0).astype(int)

# ==============================
# 3. EDA
# ==============================
print(df.head())
print(df.info())
print(df.describe())

print("\nDistribuição do target:")
print(df['default'].value_counts(normalize=True))

# ==============================
# 4. PREPARAÇÃO
# ==============================

# Missing
df.fillna(df.median(numeric_only=True), inplace=True)

X = df.drop('default', axis=1)
y = df['default']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ==============================
# 5. MODELOS
# ==============================

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)

# ==============================
# 6. AVALIAÇÃO
# ==============================

def avaliar_modelo(modelo, nome):
    y_pred_prob = modelo.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)

    ks = ks_2samp(
        y_pred_prob[y_test == 1],
        y_pred_prob[y_test == 0]
    ).statistic

    print(f"\nModelo: {nome}")
    print(f"AUC: {auc:.4f}")
    print(f"KS: {ks:.4f}")

    return y_pred_prob

prob_lr = avaliar_modelo(lr, "Logistic Regression")
prob_rf = avaliar_modelo(rf, "Random Forest")
prob_xgb = avaliar_modelo(xgb, "XGBoost")

# ==============================
# 7. FEATURE IMPORTANCE (XGBOOST)
# ==============================

importances = xgb.feature_importances_
features = X.columns

plt.figure()
plt.barh(features, importances)
plt.title("Feature Importance - XGBoost")
plt.show()

# ==============================
# 8. OTIMIZAÇÃO (GRID SEARCH)
# ==============================

param_grid = {
    'max_depth': [3, 5],
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1]
}

xgb_grid = GridSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    param_grid,
    scoring='roc_auc',
    cv=3
)

xgb_grid.fit(X_train, y_train)

print("\nMelhores parâmetros:")
print(xgb_grid.best_params_)

# Avaliação final
best_model = xgb_grid.best_estimator_
avaliar_modelo(best_model, "XGBoost Tunado")

# ==============================
# 9. CONCLUSÃO (PRINT SIMPLES)
# ==============================

print("\nProjeto finalizado com sucesso!")
print("Modelo pronto para uso em decisão de crédito.")

