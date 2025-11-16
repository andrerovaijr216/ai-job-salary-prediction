# app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Carregamento de Componentes Salvos ---
try:
    # Carregar o modelo treinado
    model_rf = joblib.load('model_rf_salary_predictor.pkl')
    # Carregar o scaler
    scaler = joblib.load('scaler.pkl')
    # Carregar a lista de colunas (features) que o modelo espera
    model_columns = joblib.load('model_columns.pkl')
    # Carregar a lista de top skills
    top_skills = joblib.load('top_skills.pkl')
    # Carregar o dataset para a Análise de Habilidades
    df_raw = pd.read_csv('ai_job_dataset.csv') 

    st.sidebar.success("Modelo e componentes carregados com sucesso!")
except FileNotFoundError as e:
    st.error(f"Erro ao carregar arquivos .pkl ou CSV: {e}. Verifique se todos os arquivos estão na mesma pasta.")
    st.stop() # Para o aplicativo se o carregamento falhar

# --- 2. Definições de Mapeamentos ---
# Mapeamentos para as variáveis categóricas ordinais
EXPERIENCE_MAP = {'Entry': 1, 'Mid': 2, 'Senior': 3, 'Executive': 4}
REMOTE_MAP = {'Presencial (0%)': 0, 'Híbrido (50%)': 50, 'Remoto (100%)': 100}
SIZE_MAP = {'Small (<50)': 'S', 'Medium (50-250)': 'M', 'Large (>250)': 'L'}

# --- 3. Função de Previsão Salarial ---
def predict_salary(input_data):
    # 1. Cria um DataFrame com todos os 0s para todas as colunas que o modelo espera
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)

    # 2. Preenche os valores numéricos
    input_df['experience_level_ordinal'] = EXPERIENCE_MAP.get(input_data['experience_level'], 0)
    input_df['years_experience'] = input_data['years_experience']
    input_df['remote_ratio'] = REMOTE_MAP.get(input_data['remote_ratio'], 0)

    # 3. Preenche as colunas One-Hot Encoding (Dummies)
    
    # Exemplo: company_size
    size_mapped = SIZE_MAP.get(input_data['company_size'], '')
    if size_mapped and f'company_size_{size_mapped}' in input_df.columns:
        input_df[f'company_size_{size_mapped}'] = 1

    # Exemplo: company_location
    location_mapped = input_data['company_location']
    if location_mapped and f'company_location_{location_mapped}' in input_df.columns:
        input_df[f'company_location_{location_mapped}'] = 1

    # Note: Você precisará mapear todos os campos que viraram dummies no treinamento!

    # 4. Preenche as colunas de Habilidades (Skills)
    for skill in input_data['selected_skills']:
        skill_col_name = f'skill_{skill.lower().replace(" ", "_")}'
        if skill_col_name in input_df.columns:
            input_df[skill_col_name] = 1

    # 5. Aplica o Scaler nas colunas numéricas originais
    numerical_cols = ['experience_level_ordinal', 'years_experience', 'remote_ratio']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # 6. Previsão
    prediction = model_rf.predict(input_df)
    return max(0, prediction[0]) # Garante que o salário previsto não seja negativo


# --- 4. Configuração da Página Streamlit ---
st.title("🌎 AI Career Navigator: Futuro do Trabalho")
st.markdown("Uma solução para medir a relevância profissional e prever salários no mercado de Inteligência Artificial/Machine Learning.")

# Cria as abas (tabs)
tab1, tab2 = st.tabs(["💰 Previsão Salarial", "🧠 Análise de Habilidades"])

# --- TAB 1: PREVISÃO SALARIAL ---
with tab1:
    st.header("Estime Seu Salário Anual em USD")
    st.markdown("Insira suas qualificações para prever o salário com base nas tendências do mercado de AI/ML.")

    # Colunas para organizar o layout de input
    col1, col2 = st.columns(2)

    with col1:
        experience_level = st.selectbox(
            "Nível de Experiência:",
            options=list(EXPERIENCE_MAP.keys()),
            help="EN (Entry), MI (Mid), SE (Senior), EX (Executive)"
        )

        years_experience = st.slider(
            "Anos de Experiência na Área:",
            min_value=0, max_value=20, value=5
        )

        company_size = st.selectbox(
            "Tamanho da Empresa:",
            options=list(SIZE_MAP.keys())
        )

    with col2:
        remote_ratio = st.selectbox(
            "Regime de Trabalho:",
            options=list(REMOTE_MAP.keys())
        )
        
        # O list de locations deve vir do dataset original (df_raw)
        company_location = st.selectbox(
            "Localização da Empresa (Top 10 para otimização):",
            options=df_raw['company_location'].value_counts().head(10).index.tolist()
        )

        # Habilidades (Multi-select)
        selected_skills = st.multiselect(
            "Selecione suas principais habilidades (Afeta o salário!):",
            options=top_skills,
            default=top_skills[:3]
        )

    # Botão de Previsão
    if st.button("CALCULAR SALÁRIO PREVISTO"):
        input_data = {
            'experience_level': experience_level,
            'years_experience': years_experience,
            'company_size': company_size,
            'remote_ratio': remote_ratio,
            'company_location': company_location,
            'selected_skills': selected_skills
        }
        
        # Chama a função de previsão
        predicted_salary = predict_salary(input_data)
        
        st.success(f"**Salário Anual Previsto:**")
        st.markdown(f"## **$ {predicted_salary:,.0f} USD**")
        st.info("Esta previsão é baseada nas tendências salariais do mercado global de AI/ML e nas suas qualificações.")

# --- TAB 2: ANÁLISE DE HABILIDADES ---
with tab2:
    st.header("Relevância de Habilidades para o Futuro")
    st.markdown("Explore as habilidades mais lucrativas e demandadas no mercado de AI/ML.")

    # Funções de Análise (Baseadas na EDA)

    # 1. Função para calcular Salário Médio por Habilidade
    def get_skill_insights(skill):
        # Filtra o dataset raw (sem as colunas dummies)
        skill_filter = df_raw['required_skills'].str.lower().str.contains(skill.lower(), na=False)
        subset = df_raw[skill_filter]
        
        count = len(subset)
        avg_salary = subset['salary_usd'].mean() if count > 0 else 0
        return count, avg_salary

    skill_to_analyze = st.selectbox(
        "Selecione uma Habilidade para Análise:",
        options=top_skills,
        index=0
    )

    if skill_to_analyze:
        count, avg_salary = get_skill_insights(skill_to_analyze)
        
        col_count, col_avg_salary = st.columns(2)

        col_count.metric(
            label=f"Vagas que exigem '{skill_to_analyze}' (Amostra)", 
            value=f"{count:,.0f}"
        )
        
        col_avg_salary.metric(
            label=f"Salário Médio Associado", 
            value=f"$ {avg_salary:,.0f} USD"
        )
        
        st.markdown("---")
        st.subheader("Sugestão de Carreira")
        
        # Lógica Simples para Sugestão de Carreira (Futuro do Trabalho)
        # Encontra o Job Title mais comum para essa habilidade
        if count > 0:
            job_title_counts = df_raw[df_raw['required_skills'].str.lower().str.contains(skill_to_analyze.lower(), na=False)]['job_title'].value_counts()
            
            st.info(f"Se você domina **{skill_to_analyze}**, o mercado de AI/ML sugere focar em:")
            st.dataframe(job_title_counts.head(5).reset_index().rename(columns={'index': 'Título do Cargo', 'job_title': 'Frequência de Ocorrência'}), 
                         hide_index=True)
        else:
            st.warning("Habilidade não encontrada ou com baixa frequência na nossa amostra.")