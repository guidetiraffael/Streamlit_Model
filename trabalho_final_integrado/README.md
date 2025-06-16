# Aplicação Interativa de Machine Learning com Streamlit e PyCaret

Este repositório contém uma aplicação interativa desenvolvida em Python, que combina o poder do Streamlit para a interface do usuário e do PyCaret para as funcionalidades de Machine Learning. O objetivo é fornecer uma ferramenta intuitiva e acessível para análise de dados, construção e avaliação de modelos de ML, e realização de previsões.

## Funcionalidades

A aplicação oferece as seguintes funcionalidades:

*   **Upload de Dados:** Carregamento fácil de arquivos CSV e Excel.
*   **Análise Exploratória de Dados (EDA):** Visualização de estatísticas descritivas, tipos de dados e valores ausentes.
*   **Seleção de Variáveis:** Escolha da variável alvo para a modelagem.
*   **Modelagem de ML:** Suporte para problemas de Classificação, Regressão e Clusterização, com treinamento e comparação de diversos modelos via PyCaret.
*   **Análise de Modelos:** Geração de gráficos e métricas para avaliação do desempenho dos modelos.
*   **Previsão:** Capacidade de realizar previsões com novos dados usando os modelos treinados.

## Arquitetura

A arquitetura do projeto é modular, seguindo um modelo cliente-servidor simplificado:

*   **Frontend (Streamlit):** Responsável pela interface gráfica do usuário, permitindo a interação com a aplicação.
*   **Backend (PyCaret):** Atua como o motor de Machine Learning, abstraindo a complexidade do ciclo de vida da modelagem.
*   **Pandas:** Utilizado para manipulação e análise de dados.

Para mais detalhes sobre a arquitetura, consulte a documentação `arquitetura.pdf` presente neste repositório.

## Como Rodar a Aplicação Localmente

Siga os passos abaixo para configurar e executar a aplicação em seu ambiente local:

### Pré-requisitos

Certifique-se de ter o Python 3.x instalado em sua máquina.

### 1. Clone o Repositório

Abra seu terminal ou prompt de comando e clone este repositório:

```bash
git clone <URL_DO_SEU_REPOSITORIO>
cd <NOME_DA_PASTA_DO_REPOSITORIO>
```

### 2. Crie e Ative um Ambiente Virtual (Recomendado)

É uma boa prática criar um ambiente virtual para isolar as dependências do projeto:

```bash
python -m venv venv
```

**No Windows:**

```bash
.\venv\Scripts\activate
```

**No macOS/Linux:**

```bash
source venv/bin/activate
```

### 3. Instale as Dependências

Com o ambiente virtual ativado, instale todas as bibliotecas necessárias usando o `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Execute a Aplicação Streamlit

Para iniciar a aplicação e acessá-la no seu navegador, utilize o seguinte comando:

```bash
python -m streamlit run app.py
```

Após executar o comando, o Streamlit abrirá automaticamente a aplicação em seu navegador padrão (geralmente em `http://localhost:8501`).

## Estrutura do Projeto

```
. 
├── app.py                     # Arquivo principal da aplicação Streamlit
├── requirements.txt           # Lista de dependências do projeto
├── arquitetura.pdf            # Documentação da arquitetura de software
├── adapters/                  # Adapters para integração com bibliotecas externas
│   ├── kaggle_downloader_adapter.py
│   ├── pycaret_adapter.py
│   ├── dtale_adapter.py
│   └── ydata_profiling_adapter.py
└── application/               # Lógica de negócio da aplicação
    └── use_cases.py
```

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.
