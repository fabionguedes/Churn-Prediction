🚀 Customer Churn Prediction - End-to-End ML Pipeline
Este projeto apresenta uma solução completa (End-to-End) de Machine Learning para prever o cancelamento de clientes (Customer Churn) em uma empresa de telecomunicações.

O foco deste repositório vai além da modelagem estatística, demonstrando fortes práticas de Engenharia de Machine Learning (MLOps), como prevenção de Data Leakage, orquestração via CLI (Command Line Interface), testes automatizados e deploy de modelo como serviço (API REST).

🔴 Teste o modelo ao vivo (Swagger UI): [CLIQUE AQUI PARA ACESSAR A API](https://churn-prediction-200a.onrender.com/docs) 

------------------------------------------------------------------------------------------------

🧠 Arquitetura do Projeto
O projeto foi desenhado para ser modular, escalável e seguro para o ambiente de produção.

Análise e Seleção de Features: Utilização do Information Value (IV) para seleção estatística das variáveis preditoras.

Prevenção de Data Leakage: Uso de ColumnTransformer e Pipeline do Scikit-Learn para encapsular o One-Hot Encoding e o algoritmo.

Modelagem: Modelo Campeão XGBoost otimizado via GridSearchCV lidando com desbalanceamento de classes (scale_pos_weight).

Orquestração: Script main.py atuando como um "Maestro" do sistema utilizando argparse.

Qualidade de Software: Testes unitários com pytest para garantir a integridade das regras de negócio.

Deploy: API servida com FastAPI e hospedada continuamente via CI/CD no Render.

📂 Estrutura do Repositório
├── data/
│   ├── raw/                      # Dados brutos originais
│   └── curated/                  # Dados limpos e processados
├── models/                       # Artefato de produção (champion_xgb_pipeline.pkl)
├── src/
│   └── a3data/
│       ├── preprocessing.py      # Lógica de limpeza e regras de negócio
│       ├── train.py              # Construção do pipeline e treinamento do XGBoost
│       ├── predict.py            # Motor de inferência para dados em Batch
│       └── api.py                # Código-fonte da API Web (FastAPI)
├── tests/
│   └── test_preprocessing.py     # Testes automatizados (Pytest)
├── main.py                       # Ponto de entrada oficial da aplicação (CLI)
├── requirements.txt              # Dependências do projeto (travadas para produção)
└── README.md

------------------------------------------------------------------------------------------------

🛠️ Como executar o projeto localmente
1. Clonar e Instalar
## Clone o repositório
git clone https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
cd SEU_REPOSITORIO

# Crie e ative um ambiente virtual (Recomendado: Python 3.11)
python -m venv .venv
source .venv/bin/activate  # No Windows use: .venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt

2. Painel de Controle (main.py)
Todo o projeto pode ser executado a partir do arquivo principal, utilizando os seguintes argumentos:
# 1. Limpeza de Dados (Gera a base curada)
python main.py --step preprocess

# 2. Treinamento do Modelo (Gera o arquivo .pkl)
python main.py --step train

# 3. Levantar o Servidor Web (Inicia a API)
python main.py --step api

3. Rodar Testes Unitários
Para garantir que as funções de pré-processamento estão seguindo as regras de negócio:
pytest tests/

------------------------------------------------------------------------------------------------

🌐 Como consumir a API (Exemplo em Python)
Com a API rodando localmente (ou usando o link público do Render), você pode fazer previsões em tempo real enviando um pacote JSON.
import requests

# URL da API (use o link do Render se estiver testando em nuvem)
url = "http://127.0.0.1:8080/predict"

# Dados do cliente
payload = {
  "customerID": "9999-AAAAA",
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 2,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": "70.70",
  "TotalCharges": "150.00"
}

response = requests.post(url, json=payload)
print(response.json())

# Saída Esperada:
# {'customerID': '9999-AAAAA', 'previsao_churn': 1, 'probabilidade_churn': 0.8245, 'mensagem': 'Alto risco de cancelamento!'}

------------------------------------------------------------------------------------------------

Autor: Fábio Guedes

LinkedIn: https://www.linkedin.com/in/fabionguedes/

Contato: fabionguedes@gmail.com
