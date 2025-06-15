import streamlit as st
import pandas as pd
import os
import sys

# Adicionar o diretório raiz do projeto ao sys.path para permitir importações relativas
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Adapters
from adapters.kaggle_downloader_adapter import KaggleDownloaderAdapter
from adapters.ydata_profiling_adapter import YDataProfilingAdapter
from adapters.dtale_adapter import DtaleAdapter
from adapters.pycaret_adapter import PyCaretAdapter

# Application
from application.use_cases import MLUseCases

# PyCaret tasks (importar diretamente para usar save_model e load_model)
from pycaret.classification import setup as class_setup, compare_models as class_compare, pull as class_pull, save_model as class_save_model, load_model as class_load_model, plot_model as class_plot_model
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, pull as reg_pull, save_model as reg_save_model, load_model as reg_load_model, plot_model as reg_plot_model
from pycaret.clustering import setup as clus_setup, create_model as clus_create, assign_model as clus_assign, pull as clus_pull, save_model as clus_save_model, load_model as clus_load_model, plot_model as clus_plot_model

st.set_page_config(layout="wide")
st.title("Aplicação de Machine Learning com Streamlit e PyCaret (Integrado)")

# Instantiate adapters
kaggle_adapter = KaggleDownloaderAdapter()
profiler_adapter = YDataProfilingAdapter()
dtale_adapter = DtaleAdapter()
training_adapter = PyCaretAdapter()

# Create the use-case orchestrator with the chosen adapters
ml_use_cases = MLUseCases(
    dataset_adapter=kaggle_adapter,
    profiler_adapter=profiler_adapter,
    dtale_adapter=dtale_adapter,
    training_adapter=training_adapter
)

# Função para carregar dados
@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file)
    else:
        st.error("Formato de arquivo não suportado. Por favor, carregue um arquivo CSV ou Excel.")
        return None
    return df

# Upload de dados
st.sidebar.header("Upload de Dados")
uploaded_file = st.sidebar.file_uploader("Escolha um arquivo CSV ou Excel", type=["csv", "xls", "xlsx"])

if uploaded_file is not None:
    data = load_data(uploaded_file)
    if data is not None:
        st.success("Dados carregados com sucesso!")
        st.write("Prévia dos dados:", data.head())

        st.sidebar.header("Análise Exploratória de Dados (EDA)")
        if st.sidebar.checkbox("Mostrar estatísticas descritivas"):
            st.subheader("Estatísticas Descritivas")
            st.write(data.describe())
        if st.sidebar.checkbox("Mostrar tipos de dados"):
            st.subheader("Tipos de Dados")
            st.write(data.dtypes)
        if st.sidebar.checkbox("Mostrar valores ausentes"):
            st.subheader("Valores Ausentes")
            st.write(data.isnull().sum())

        st.sidebar.header("Seleção de Variáveis")
        all_columns = data.columns.tolist()
        target_column = st.sidebar.selectbox("Selecione a variável alvo (para Classificação/Regressão)", all_columns)

        problem_type = st.sidebar.radio("Selecione o tipo de problema", ("Classificação", "Regressão", "Clusterização"))

        model_path = "best_model"

        if st.sidebar.button("Executar PyCaret"):
            if problem_type == "Classificação":
                if target_column:
                    st.subheader("Executando Classificação com PyCaret")
                    with st.spinner("Configurando ambiente PyCaret para Classificação..."):
                        class_setup(data, target=target_column, session_id=123, silent=True, verbose=False)
                    st.success("Ambiente de Classificação configurado.")
                    with st.spinner("Comparando modelos de Classificação..."):
                        best_model = class_compare()
                    st.subheader("Melhor Modelo de Classificação")
                    st.write(class_pull())
                    class_save_model(best_model, model_path)
                    st.success("Modelos de Classificação comparados e melhor modelo salvo.")

                    st.subheader("Análise do Modelo de Classificação")
                    plot_type = st.selectbox("Selecione o tipo de plot para Classificação", ["auc", "confusion_matrix", "precision_recall", "error", "boundary"])
                    try:
                        class_plot_model(best_model, plot=plot_type, save=True)
                        st.image(f'{plot_type}.png')
                    except Exception as e:
                        st.error(f"Não foi possível gerar o plot {plot_type}: {e}")

                else:
                    st.warning("Por favor, selecione uma variável alvo para Classificação.")

            elif problem_type == "Regressão":
                if target_column:
                    st.subheader("Executando Regressão com PyCaret")
                    with st.spinner("Configurando ambiente PyCaret para Regressão..."):
                        reg_setup(data, target=target_column, session_id=123, silent=True, verbose=False)
                    st.success("Ambiente de Regressão configurado.")
                    with st.spinner("Comparando modelos de Regressão..."):
                        best_model = reg_compare()
                    st.subheader("Melhor Modelo de Regressão")
                    st.write(reg_pull())
                    reg_save_model(best_model, model_path)
                    st.success("Modelos de Regressão comparados e melhor modelo salvo.")

                    st.subheader("Análise do Modelo de Regressão")
                    plot_type = st.selectbox("Selecione o tipo de plot para Regressão", ["residuals", "error", "cooks", "learning", "vc", "manifold", "feature", "rfe", "tree"])
                    try:
                        reg_plot_model(best_model, plot=plot_type, save=True)
                        st.image(f'{plot_type}.png')
                    except Exception as e:
                        st.error(f"Não foi possível gerar o plot {plot_type}: {e}")

                else:
                    st.warning("Por favor, selecione uma variável alvo para Regressão.")

            elif problem_type == "Clusterização":
                st.subheader("Executando Clusterização com PyCaret")
                with st.spinner("Configurando ambiente PyCaret para Clusterização..."):
                    clus_setup(data, session_id=123, silent=True, verbose=False)
                st.success("Ambiente de Clusterização configurado.")
                num_clusters = st.number_input("Número de Clusters (K-Means)", min_value=2, value=3)
                if st.button("Criar Modelo de Clusterização"):
                    with st.spinner(f"Criando modelo K-Means com {num_clusters} clusters..."):
                        kmeans = clus_create("kmeans", num_clusters=num_clusters)
                        kmeans_results = clus_assign(kmeans)
                    st.subheader("Resultados da Clusterização (K-Means)")
                    st.write(kmeans_results.head())
                    clus_save_model(kmeans, model_path)
                    st.success("Modelo de Clusterização criado, resultados atribuídos e modelo salvo.")

                    st.subheader("Análise do Modelo de Clusterização")
                    plot_type = st.selectbox("Selecione o tipo de plot para Clusterização", ["elbow", "silhouette", "distance", "distribution"])
                    try:
                        clus_plot_model(kmeans, plot=plot_type, save=True)
                        st.image(f'{plot_type}.png')
                    except Exception as e:
                        st.error(f"Não foi possível gerar o plot {plot_type}: {e}")

        st.sidebar.header("Previsão com Novos Dados")
        new_data_file = st.sidebar.file_uploader("Upload de novos dados para previsão (CSV ou Excel)", type=["csv", "xls", "xlsx"])
        if new_data_file is not None:
            new_data = load_data(new_data_file)
            if new_data is not None:
                st.success("Novos dados carregados com sucesso!")
                st.write("Prévia dos novos dados:", new_data.head())
                if st.sidebar.button("Fazer Previsão"):
                    if os.path.exists(f"{model_path}.pkl"):
                        if problem_type == "Classificação":
                            loaded_model = class_load_model(model_path)
                            predictions = class_predict_model(loaded_model, data=new_data)
                            st.subheader("Previsões de Classificação")
                            st.write(predictions.head())
                        elif problem_type == "Regressão":
                            loaded_model = reg_load_model(model_path)
                            predictions = reg_predict_model(loaded_model, data=new_data)
                            st.subheader("Previsões de Regressão")
                            st.write(predictions.head())
                        elif problem_type == "Clusterização":
                            st.warning("Previsão para clusterização não é um conceito direto como classificação/regressão. Os dados são atribuídos a clusters existentes.")
                            st.write("Para ver a atribuição de clusters, use a opção 'Criar Modelo de Clusterização' novamente com os novos dados.")
                    else:
                        st.warning("Nenhum modelo treinado encontrado. Por favor, execute o PyCaret primeiro.")

else:
    st.info("Por favor, faça o upload de um arquivo para começar.")


