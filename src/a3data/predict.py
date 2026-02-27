import os
import pandas as pd
import joblib

# Importando as funções que criamos na Fase 1! 
# (Ajuste o import 'src.preprocessing' se a sua estrutura de pastas for diferente)
from src.a3data.preprocessing import clean_numeric_features, drop_low_iv_features

def load_production_model(model_path: str):
    """Carrega o modelo treinado (pipeline) do disco."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado em: {model_path}")
    return joblib.load(model_path)

def make_prediction(raw_customer_data: pd.DataFrame, model) -> pd.DataFrame:
    """
    Recebe os dados brutos de novos clientes, aplica a limpeza básica 
    e retorna o DataFrame com as previsões e probabilidades de Churn.
    """
    # 1. Limpeza básica reutilizando nossas funções de produção
    df_clean = clean_numeric_features(raw_customer_data)
    df_clean = drop_low_iv_features(df_clean)
    
    # 2. Fazendo a previsão usando o Pipeline (que já faz o One-Hot Encoding)
    # Nota: Não precisamos do encode_target() aqui, pois em produção não sabemos o alvo (Churn)
    predictions = model.predict(df_clean)
    probabilities = model.predict_proba(df_clean)[:, 1] # Pegando apenas a probabilidade da classe 1 (Yes)
    
    # 3. Montando o resultado
    result_df = raw_customer_data.copy()
    result_df['Predict_Churn'] = predictions
    result_df['Churn_Probability'] = np.round(probabilities, 4)
    
    return result_df

# --- SIMULANDO O USO EM PRODUÇÃO ---
if __name__ == "__main__":
    import numpy as np # Importado aqui apenas para o arredondamento na simulação
    
    # Caminho do modelo
    MODEL_PATH = '/Users/guedesf/Documents/Data Science/Projetos/a3data/models/champion_xgb_pipeline.pkl'
    
    # Simulando a chegada de dados de 2 clientes novos (em formato bruto, como viria do banco)
    novos_clientes = pd.DataFrame([
        {
            'customerID': '9999-AAAAA', 'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'Yes', 'Dependents': 'No',
            'tenure': 2, 'PhoneService': 'Yes', 'MultipleLines': 'No', 'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No', 'OnlineBackup': 'No', 'DeviceProtection': 'No', 'TechSupport': 'No',
            'StreamingTV': 'No', 'StreamingMovies': 'No', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check', 'MonthlyCharges': '70,70', 'TotalCharges': '150,00'
        },
        {
            'customerID': '8888-BBBBB', 'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
            'tenure': 65, 'PhoneService': 'Yes', 'MultipleLines': 'Yes', 'InternetService': 'DSL',
            'OnlineSecurity': 'Yes', 'OnlineBackup': 'Yes', 'DeviceProtection': 'Yes', 'TechSupport': 'Yes',
            'StreamingTV': 'Yes', 'StreamingMovies': 'Yes', 'Contract': 'Two year', 'PaperlessBilling': 'No',
            'PaymentMethod': 'Credit card (automatic)', 'MonthlyCharges': '90.00', 'TotalCharges': '5800.00'
        }
    ])
    
    print("Carregando o modelo de produção...")
    modelo_em_producao = load_production_model(MODEL_PATH)
    
    print("Processando e prevendo novos clientes...")
    df_resultados = make_prediction(novos_clientes, modelo_em_producao)
    
    print("\n--- RESULTADOS DA INFERÊNCIA ---")
    # Mostrando apenas as colunas principais para conferência
    print(df_resultados[['customerID', 'Contract', 'tenure', 'Predict_Churn', 'Churn_Probability']])