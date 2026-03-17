# 📊 Credit Score com Machine Learning

## 🎯 Objetivo de Negócio

Desenvolver um modelo de **credit score** para prever a probabilidade de inadimplência de clientes, apoiando decisões de concessão de crédito, definição de limites e estratégias de risco.

---

## 🧠 Contexto

Instituições financeiras utilizam modelos estatísticos e de machine learning para estimar o risco de crédito dos clientes. Neste projeto, foi construída uma solução completa, desde a análise exploratória até a modelagem e avaliação, simulando um cenário real de decisão de crédito.

---

## 📂 Estrutura do Projeto

```
credit_score_ml/
├── data/
├── notebooks/
├── src/
├── models/
├── README.md
```

---

## 📊 Dados Utilizados

O dataset contém informações comportamentais e cadastrais dos clientes, como:

* Idade
* Renda
* Tempo de emprego
* Quantidade de atrasos
* Utilização de limite de crédito

**Variável target:**

* `default` → Indica inadimplência (1 = inadimplente, 0 = adimplente)

---

## 🔍 Análise Exploratória (EDA)

Foram realizadas análises para:

* Identificar distribuição da variável target
* Avaliar presença de valores ausentes
* Detectar possíveis outliers
* Entender relação entre variáveis e inadimplência

---

## ⚙️ Metodologia

### 🔹 Pré-processamento

* Tratamento de valores ausentes
* Separação entre variáveis explicativas (X) e target (y)
* Divisão treino/teste (70/30)

---

### 🤖 Modelos Utilizados

* **Regressão Logística** (baseline)
* **Random Forest**
* **XGBoost** (modelo principal)

---

### 📈 Métricas de Avaliação

* **AUC (Area Under the Curve)**
  Mede a capacidade do modelo de separar bons e maus pagadores

* **KS (Kolmogorov-Smirnov)**
  Métrica amplamente utilizada em risco de crédito para avaliar separação entre distribuições

---

## 🏆 Resultados

| Modelo              | AUC  | KS   |
| ------------------- | ---- | ---- |
| Logistic Regression | 0.XX | 0.XX |
| Random Forest       | 0.XX | 0.XX |
| XGBoost             | 0.XX | 0.XX |

✅ O modelo **XGBoost** apresentou a melhor performance geral.

---

## 🔧 Otimização do Modelo

Foi aplicado **Grid Search** para ajuste de hiperparâmetros do XGBoost, melhorando a performance do modelo.

---

## 📊 Importância das Variáveis

As variáveis mais relevantes para previsão de inadimplência foram:

* Quantidade de atrasos
* Utilização do limite
* Renda

---

## 💡 Principais Insights de Negócio

* Clientes com maior utilização de limite apresentam maior risco de inadimplência
* Histórico de atrasos é um dos principais preditores de default
* Modelos de machine learning apresentam ganho relevante em relação à regressão logística

---

## 🚀 Aplicações Práticas

Este modelo pode ser utilizado para:

* Aprovação ou rejeição de crédito
* Definição de limites
* Segmentação de clientes por risco
* Precificação (risk-based pricing)

---

## 📌 Próximos Passos

* Implementação de **LGD e EAD** para cálculo de perda esperada
* Monitoramento de performance (model drift)
* Explicabilidade com SHAP
* Deploy do modelo em ambiente produtivo

---

## 🧑‍💻 Autor

Murilo Anastácio Machado

---

## 🏁 Conclusão

Este projeto demonstra a aplicação prática de técnicas de machine learning em risco de crédito, com foco em geração de valor para o negócio e tomada de decisão orientada a dados.
