# 👨‍💻 Global Solution - FIAP | AI Career Navigator: O Futuro do Trabalho

**Disciplina:** Front End & Mobile Development
**Turma:** 2TIAPY
**Prazo de Entrega:** 19/11/2025
**Local de Entrega:** Portal FIAP

## 🔗 Links do Projeto

| Recurso | Link | Observações |
| :--- | :--- | :--- |
| **Webapp Deployado** | **[https://aicareernavigator.streamlit.app/](https://aicareernavigator.streamlit.app/)** | **Entrega 3:** Webapp funcional na nuvem do Streamlit. |
| **Repositório Github** | **[https://github.com/andrerovaijr216/ai-job-salary-prediction](https://github.com/andrerovaijr216/ai-job-salary-prediction)** | **Entrega 2:** Contém todos os códigos e modelos. |
| **Notebook (Google Colab)** | **[https://colab.research.google.com/drive/1ajfcBeiiXS3a1WyTtrgzwZAqmlc7Ga_a?usp=sharing](https://colab.research.google.com/drive/1ajfcBeiiXS3a1WyTtrgzwZAqmlc7Ga_a?usp=sharing)** | Versão executável do Notebook (`.ipynb`). |

---

## 👥 Integrantes

| Nome Completo | RM |
| :--- | :--- |
| André Rovai | RM555848 |
| Alan de Souza | RM557088 |
| Leonardo Zago | RM558691 |

---

## 💡 Descrição Detalhada do Projeto (Requisito PDF 1)

Este projeto implementa o **AI Career Navigator**, um webapp interativo desenvolvido em **Streamlit** com um modelo de **Machine Learning (Regressão)** embarcado, aplicado ao contexto de **"O Futuro do Trabalho"** no setor de Inteligência Artificial e Machine Learning.

### 1.1. Motivação do Projeto (Requisito PDF 1.1)

O mercado de trabalho de AI/ML cresce exponencialmente, mas é marcado por uma grande dispersão salarial e rápida obsolescência de *skills*. A motivação é combater a incerteza profissional, fornecendo aos usuários uma ferramenta de **orientação de carreira baseada em dados** para:

*   Medir a **relevância** de suas habilidades atuais em relação às mais bem pagas do futuro.
*   Estimular a aquisição de *skills* com maior potencial de retorno financeiro.

### 1.2. Objetivo (Requisito PDF 1.2)

O objetivo principal é criar um webapp que utilize dados e Machine Learning para auxiliar o usuário na medição de sua empregabilidade e potencial salarial no mercado de AI/ML. As funcionalidades-chave são:

1.  **Previsão Salarial:** Estimar o salário anual (em USD) para um perfil de vaga ou candidato, considerando fatores como Nível de Experiência, Localização e Habilidades (Modelo de Regressão).
2.  **Análise de Habilidades:** Apresentar a frequência e o salário médio de mercado associados às principais habilidades (Futuro do Trabalho/Relevância).
3.  **Orientação de Carreira:** Sugerir títulos de cargo (Job Titles) com alta demanda e remuneração, que se alinham com um conjunto de habilidades específicas.

### 1.3. Resultados Esperados (Requisito PDF 1.3)

O projeto entregou os seguintes componentes:

1.  **Notebook Completo:** Implementação do pipeline de Data Science (Carregamento, Limpeza, EDA e Modelagem).
2.  **Modelo de Regressão:** Um modelo **Random Forest Regressor** treinado com o dataset *Global AI Job Market & Salary Trends 2025*.
    *   **Métricas de Avaliação:** **R² Score** de **0.5864** e **MAE** (Erro Absoluto Médio) de **$ 28,586.90 USD** (Valor aceitável para um dataset sintético com alta variabilidade).
3.  **Webapp em Streamlit:** Aplicação interativa `app.py` que consome o modelo e apresenta as análises de *skills*.
4.  **Deploy na Nuvem:** Aplicação acessível publicamente na Streamlit Cloud.

---

## 🏗️ Estrutura e Desenvolvimento (Requisitos do Projeto)

O projeto foi estruturado para atender rigorosamente aos 3 pontos obrigatórios do GS:

### 1. Notebook `.ipynb` (Requisito de Avaliação 2: 3 pontos)

| Seção | Descrição | Status |
| :--- | :--- | :--- |
| **1.1. Carregamento e Limpeza** | Leitura do `ai_job_dataset.csv`. Tratamento de nulos (`fillna` em `required_skills`) e padronização de variáveis categóricas (`company_size`, `experience_level`). | ✅ Completo |
| **1.2. Análise Exploratória (EDA)** | Análise de distribuição salarial, relação Salário vs. Experiência/Tamanho da Empresa, e *Feature Engineering* para ranking de **Top 50 Habilidades** mais demandadas. | ✅ Completo |
| **1.3. Modelagem (Machine Learning)** | Criação de *Dummies* para variáveis categóricas (One-Hot Encoding), Escalonamento (`StandardScaler`), separação Treino/Teste (80/20) e treinamento do **Random Forest Regressor**. Salvamento dos objetos `.pkl`. | ✅ Completo |

### 2. Desenvolvimento do Webapp (Requisito de Avaliação 3: 4 pontos)

*   **Tecnologia:** Streamlit (`app.py`).
*   **Modelo Embarcado:** O `app.py` carrega o modelo, o scaler e as colunas do arquivo `.pkl` para fazer a previsão.
*   **Descompactação:** Foi implementada uma função (`setup_files()`) com `@st.cache_resource` para descompactar o arquivo `assets.zip` no início, garantindo o funcionamento no ambiente de deploy.
*   **Funcionalidades:**
    *   **Previsão Salarial:** Formulário interativo para entrada de dados e *output* da previsão em tempo real.
    *   **Análise de Habilidades:** Aba dedicada a mostrar a frequência e o salário médio de *skills* selecionadas.

### 3. Deploy do Webapp (Requisito de Avaliação 4: 1 ponto extra)

*   **Plataforma:** Streamlit Community Cloud.
*   **Arquivos na Raiz:** `app.py`, `requirements.txt`, `assets.zip` (contendo todos os `.pkl` e o CSV).
*   **Status:** Aplicação em produção no link fornecido.

---

## 🛠️ Detalhes Técnicos e Arquivos

| Nome do Arquivo | Conteúdo | Uso |
| :--- | :--- | :--- |
| `app.py` | Código Python do Webapp Streamlit. | **Motor da Aplicação.** |
| `Global_Solution_Futuro_do_Trabalho.ipynb` | Notebook completo do projeto. | **Entregável Notebook.** |
| `requirements.txt` | Lista de dependências (streamlit, scikit-learn, etc.). | **Requisito de Deploy.** |
| `assets.zip` | Contém: `model_rf_salary_predictor.pkl`, `scaler.pkl`, `model_columns.pkl`, `top_skills.pkl`, `ai_job_dataset.csv`. | **Componentes do Modelo e Dados.** |
