# app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import zipfile
import os
import shutil # Adicionado para garantir a cópia de objetos, se necessário

# --- 1. SETUP INICIAL: DESCOMPACTAÇÃO DE ARQUIVOS ---
# **IMPORTANTE:** O nome do arquivo ZIP no seu GitHub deve ser 'assets.zip'
ZIP_FILE_NAME = 'assets.zip' 

# Lista de arquivos essenciais que devem existir após a descompactação
ESSENTIAL_FILES = [
    'model_rf_salary_predictor.pkl',
    'scaler.pkl',
    'model_columns.pkl',
    'top_skills.pkl',
    'ai_job_dataset.csv'
]

@st.cache_resource
def setup_files():
    """Verifica e extrai arquivos essenciais de um ZIP se necessário."""
    
    # 1. Verifica se todos os arquivos essenciais já existem
    if all(os.path.exists(f) for f in ESSENTIAL_FILES):
        return True

    # 2. Tenta extrair do ZIP
    elif os.path.exists(ZIP_FILE_NAME):
        try:
            # st.spinner é uma boa prática para dar feedback ao usuário
            with st.spinner(f"Modelos não encontrados. Descompactando '{ZIP_FILE_NAME}'..."):
                with zipfile.ZipFile(ZIP_FILE_NAME, 'r') as zip_ref:
                    # Tenta a extração padrão, que funciona se não houver subpastas
                    zip_ref.extractall('.') 

            # Confirma que a extração resultou nos arquivos esperados
            if all(os.path.exists(f) for f in ESSENTIAL_FILES):
                return True
            else:
                # Caso a extração padrão falhe por subpasta, exibe um erro
                st.error("Falha ao encontrar arquivos após descompactação. Verifique se o ZIP não contém subpastas internas (ex: 'assets/arquivo.pkl').")
                return False
        
        except Exception as e:
            st.error(f"Erro ao descompactar '{ZIP_FILE_NAME}': {e}")
            return False
    
    # 3. Falha se nem arquivos nem ZIP são encontrados
    else:
        st.error(f"Erro Crítico: Arquivos de modelo e o backup '{ZIP_FILE_NAME}' não foram encontrados. O deploy falhará.")
        return False

# Execução do Setup: A aplicação SÓ CONTINUA se os arquivos estiverem prontos.
if not setup_files():
    st.stop()
    
# --- 2. CARREGAMENTO DOS COMPONENTES (COM CACHE) ---

@st.cache_resource
def load_components():
    """Carrega todos os modelos e dados após a descompactação."""
    model = joblib.load(ESSENTIAL_FILES[0])
    scaler = joblib.load(ESSENTIAL_FILES[1])
    model_cols = joblib.load(ESSENTIAL_FILES[2])
    top_skills = joblib.load(ESSENTIAL_FILES[3])
    df = pd.read_csv(ESSENTIAL_FILES[4])
    return model, scaler, model_cols, top_skills, df

try:
    model_rf, scaler, model_columns, top_skills, df_raw = load_components()
    st.sidebar.success("Modelo e componentes carregados com sucesso!")
except Exception as e:
    # Este bloco é uma salvaguarda, pois setup_files() deve ter pego o erro
    st.error(f"Erro no carregamento após descompactação: {e}. Verifique o conteúdo do seu ZIP.")
    st.stop()


# --- 3. Definições de Mapeamentos ---
EXPERIENCE_MAP = {'Entry': 1, 'Mid': 2, 'Senior': 3, 'Executive': 4}
REMOTE_MAP = {'Presencial (0%)': 0, 'Híbrido (50%)': 50, 'Remoto (100%)': 100}
SIZE_MAP = {'Small (<50)': 'S', 'Medium (50-250)': 'M', 'Large (>250)': 'L'}


# --- 4. Função de Previsão Salarial ---
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
        
    # **IMPORTANTE:** Mapeamento de Job Title Dummies
    # O modelo espera uma coluna com o job_title selecionado. 
    job_title_mapped = input_data.get('job_title', '') # Adicione 'job_title' no input_data se for usar
    if job_title_mapped and f'job_title_{job_title_mapped}' in input_df.columns:
        input_df[f'job_title_{job_title_mapped}'] = 1
        
    # **IMPORTANTE:** Mapeamento de Education Dummies
    # O modelo espera uma coluna com a education_required selecionada.
    education_mapped = input_data.get('education_required', '') # Adicione 'education_required' no input_data se for usar
    if education_mapped and f'education_required_{education_mapped}' in input_df.columns:
        input_df[f'education_required_{education_mapped}'] = 1


    # 4. Preenche as colunas de Habilidades (Skills)
    for skill in input_data['selected_skills']:
        skill_col_name = f'skill_{skill.lower().replace(" ", "_")}'
        if skill_col_name in input_df.columns:
            input_df[skill_col_name] = 1

    # 5. Aplica o Scaler nas colunas numéricas originais
    numerical_cols = ['experience_level_ordinal', 'years_experience', 'remote_ratio']
    # O scikit-learn espera um array 2D, mesmo que seja apenas uma linha
    # Verifica se os dados numéricos são válidos antes de transformar
    try:
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols].values.reshape(1, -1))
    except Exception as e:
        st.error(f"Erro no pré-processamento de dados: {e}")
        return 0

    # 6. Previsão
    prediction = model_rf.predict(input_df)
    return max(0, prediction[0])


# --- 6. Configuração da Página Streamlit (Interface) ---
st.title("🌎 AI Career Navigator: Futuro do Trabalho")
st.markdown("Uma solução para medir a relevância profissional e prever salários no mercado de Inteligência Artificial/Machine Learning.")

# Cria as abas (tabs)
tab1, tab2 = st.tabs(["💰 Previsão Salarial", "🧠 Análise de Habilidades"])

# --- TAB 1: PREVISÃO SALARIAL ---
with tab1:
    st.header("Estime Seu Salário Anual em USD")
    st.markdown("Insira suas qualificações para prever o salário com base nas tendências do mercado de AI/ML.")

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)


    # Adicionado Job Title e Education Required para a funcionalidade completa do modelo
    with col3:
        job_title = st.selectbox(
            "Seu Cargo (ou Desejado):",
            options=df_raw['job_title'].value_counts().head(20).index.tolist() # Top 20 para simplificar
        )

    with col4:
        education_required = st.selectbox(
            "Nível de Educação:",
            options=df_raw['education_required'].unique().tolist()
        )

    with col1:
        experience_level = st.selectbox(
            "Nível de Experiência:",
            options=list(EXPERIENCE_MAP.keys())
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
        
        top_locations = df_raw['company_location'].value_counts().head(10).index.tolist()
        company_location = st.selectbox(
            "Localização da Empresa (Top 10):",
            options=top_locations
        )
        
        # O multiselect de skills está na coluna 2 para melhor layout

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
            'selected_skills': selected_skills,
            'job_title': job_title, # Adicionado
            'education_required': education_required # Adicionado
        }
        
        predicted_salary = predict_salary(input_data)
        
        st.success(f"**Salário Anual Previsto:**")
        st.markdown(f"## **$ {predicted_salary:,.0f} USD**")
        st.info("Esta previsão é baseada nas tendências salariais do mercado global de AI/ML e nas suas qualificações.")

# --- TAB 2: ANÁLISE DE HABILIDADES ---
with tab2:
    st.header("Relevância de Habilidades para o Futuro")
    st.markdown("Explore as habilidades mais lucrativas e demandadas no mercado de AI/ML.")

    @st.cache_data
    def get_skill_insights(df, skill):
        # Filtra o dataset raw (sem as colunas dummies)
        skill_filter = df['required_skills'].str.lower().str.contains(skill.lower(), na=False)
        subset = df[skill_filter]
        
        count = len(subset)
        avg_salary = subset['salary_usd'].mean() if count > 0 else 0
        return count, avg_salary

    skill_to_analyze = st.selectbox(
        "Selecione uma Habilidade para Análise:",
        options=top_skills,
        index=0
    )

    if skill_to_analyze:
        count, avg_salary = get_skill_insights(df_raw, skill_to_analyze)
        
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
        
        if count > 0:
            # Uso de cache_data para a contagem de Job Titles
            @st.cache_data
            def get_job_title_counts(df, skill):
                return df[df['required_skills'].str.lower().str.contains(skill.lower(), na=False)]['job_title'].value_counts()
            
            job_title_counts = get_job_title_counts(df_raw, skill_to_analyze)

            st.info(f"Se você domina **{skill_to_analyze}**, o mercado de AI/ML sugere focar em:")
            st.dataframe(job_title_counts.head(5).reset_index().rename(columns={'index': 'Título do Cargo', 'job_title': 'Frequência de Ocorrência'}), 
                         hide_index=True)
        else:
            st.warning("Habilidade não encontrada ou com baixa frequência na nossa amostra.")
