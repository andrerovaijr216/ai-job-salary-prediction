# 👨‍💻 Global Solution - FIAP | AI Career Navigator: O Futuro do Trabalho

**Disciplina:** Front End & Mobile Development
**Turma:** 2TIAPY
**Prazo de Entrega:** 19/11/2025

## 👥 Integrantes
| Nome Completo | RM |
| :--- | :--- |
| André Rovai | RM555848 |
| Alan de Souza | RM557088 |
| Leonardo Zago | RM558691 |

## 💡 Descrição do Projeto (O Futuro do Trabalho)

Este projeto é um webapp interativo desenvolvido em Streamlit com um modelo de Machine Learning (Regressão) embarcado, aplicado ao contexto de "O Futuro do Trabalho", com foco no crescente mercado de Inteligência Artificial (AI) e Machine Learning (ML).

### 1. Motivação do Projeto

O mercado de trabalho está em constante e rápida transformação, impulsionado pela AI. A falta de clareza sobre quais habilidades e profissões serão mais relevantes no futuro gera incerteza profissional. A motivação é fornecer uma ferramenta de **orientação de carreira baseada em dados**, permitindo que o usuário entenda a relevância de suas habilidades e visualize o potencial salarial em diferentes segmentos da AI.

### 2. Objetivo

Desenvolver um webapp que permita ao usuário:
1.  **Prever o Salário** (em USD) com alta precisão, baseado em suas características (experiência, habilidades, localização) usando um modelo de Machine Learning.
2.  **Analisar a Relevância de Habilidades** (Futuro do Trabalho), mostrando o impacto de habilidades específicas (como TensorFlow, PyTorch, etc.) na demanda e no salário médio de mercado.
3.  **Auxiliar na Transição de Carreira**, identificando os títulos de trabalho mais lucrativos para um determinado conjunto de habilidades.

### 3. Resultados Esperados

1.  **Notebook (.ipynb):** Completo com as etapas de Carregamento, Limpeza, Análise Exploratória (EDA) e Modelagem (Random Forest Regressor).
2.  **Webapp (Streamlit):** Interface intuitiva e funcional que carrega o modelo treinado para predição em tempo real.
3.  **Deploy:** Aplicação disponível publicamente no Streamlit Community Cloud.
4.  **Modelo de ML:** Um modelo de Regressão com bom desempenho (R² e MAE) na previsão salarial.

---

## 🛠️ Tecnologias e Arquivos

*   **Linguagem:** Python
*   **Webapp:** Streamlit
*   **Machine Learning:** Scikit-Learn (Random Forest Regressor)
*   **Dados:** Pandas, NumPy
*   **Dataset:** `ai_job_dataset.csv` (Global AI Job Market & Salary Trends)

## 🔗 Links

**Link do Webapp "Deployado":** [INSIRA O LINK DO STREAMLIT AQUI APÓS O DEPLOY]

**Link do Repositório Github:** [ESTE REPOSITÓRIO]

**Observação:** O notebook de treinamento foi dividido nas seções 1, 2 e 3 e resultou nos seguintes arquivos essenciais para o webapp: `model_rf_salary_predictor.pkl`, `scaler.pkl`, `model_columns.pkl` e `top_skills.pkl`.
