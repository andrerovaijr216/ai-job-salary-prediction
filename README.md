# Global Solution - FIAP: Análise Automática de Imagens com IA

## 📝 Descrição do Projeto

Este projeto, desenvolvido para a disciplina de Front End & Mobile Development da FIAP, é um sistema de interpretação automática de imagens que utiliza modelos multimodais de Inteligência Artificial. A solução é capaz de analisar uma imagem de um ambiente de trabalho e realizar duas funções principais:

1.  **Geração de Descrição Textual:** Cria uma descrição rica e detalhada da cena, identificando objetos, ações e a atmosfera do ambiente, de forma similar à percepção humana.
2.  **Extração de Informações Estruturadas:** Detecta e lista objetos específicos, reconhece pessoas e lê textos presentes na imagem (OCR).

O objetivo é demonstrar a aplicação prática de modelos de visão computacional e linguagem para a compreensão profunda de cenários do mundo profissional.

## 👥 Integrantes do Grupo

| Nome               | RM       |
| ------------------ | -------- |
| André Rovai        | RM555848 |
| Alan de Souza      | RM557088 |
| Leonardo Zago      | RM558691 |

## 📂 Estrutura de Arquivos

A imagem abaixo mostra a organização dos principais arquivos do projeto:

![Estrutura de Arquivos do Projeto](input_file_0.png)

-   `app.py`: O arquivo principal da aplicação (provavelmente construído com Streamlit ou Flask).
-   `ai_job_dataset.csv`: Dataset utilizado para o treinamento ou análise relacionada ao projeto.
-   `requirements.txt`: Lista de dependências Python necessárias para executar o projeto.
-   `*.pkl`: Arquivos de modelo serializados (pickle), contendo o modelo de Machine Learning treinado, o scaler e outras configurações.
-   `venv/`: Pasta do ambiente virtual Python (não incluída no repositório).

## 🚀 Como Executar o Projeto

Siga os passos abaixo para configurar e rodar a aplicação localmente.

1.  **Clone o repositório:**
    ```bash
    git clone [URL-DO-SEU-REPOSITORIO]
    cd [NOME-DO-SEU-REPOSITORIO]
    ```

2.  **Crie e ative um ambiente virtual:**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Execute a aplicação:**
    ```bash
    # Se for um app Streamlit
    streamlit run app.py

    # Se for um app Flask
    python app.py
    ```

5.  Abra o navegador e acesse o endereço fornecido no terminal (geralmente `http://localhost:8501` para Streamlit ou `http://localhost:5000` para Flask).

## 🛠️ Tecnologias Utilizadas

-   **Linguagem:** Python
-   **Framework Web:** Streamlit / Flask (a ser confirmado)
-   **Machine Learning:** Scikit-learn, Pandas, NumPy
-   **Visão Computacional:** OpenCV, Ultralytics (YOLO), EasyOCR
-   **Modelos Multimodais:** OpenAI GPT-4o / Google Gemini
