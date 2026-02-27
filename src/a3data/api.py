from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os

# Importando nossas funções de produção
from src.a3data.preprocessing import clean_numeric_features, drop_low_iv_features

# Inicializa o app FastAPI
app = FastAPI(
    title="API de Previsão de Churn",
    description="API para prever a probabilidade de cancelamento de clientes usando XGBoost.",
    version="1.0"
)

# Caminho do modelo (ajuste se necessário)
# Descobre a pasta raiz do projeto de forma dinâmica
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Monta o caminho para a pasta models independentemente de onde o código está rodando
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'champion_xgb_pipeline.pkl')

# Variável global para armazenar o modelo na memória quando a API iniciar
modelo_producao = None

@app.on_event("startup")
def load_model():
    """Carrega o modelo para a memória quando o servidor ligar."""
    global modelo_producao
    if os.path.exists(MODEL_PATH):
        modelo_producao = joblib.load(MODEL_PATH)
        print("Modelo XGBoost carregado com sucesso!")
    else:
        print(f"ALERTA: Modelo não encontrado no caminho {MODEL_PATH}")

# Definindo o contrato de dados de entrada (Schema)
class ClienteInput(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: str
    TotalCharges: str

@app.post("/predict")
def predict_churn(cliente: ClienteInput):
    """
    Recebe os dados de um cliente, processa e retorna a probabilidade de Churn.
    """
    if modelo_producao is None:
        raise HTTPException(status_code=500, detail="Modelo não está carregado no servidor.")
    
    # Converte o input (JSON/Pydantic) em um DataFrame do Pandas (1 linha)
    df_raw = pd.DataFrame([cliente.dict()])
    
    try:
        # Aplica o pipeline de limpeza (nossas funções da Fase 1)
        df_clean = clean_numeric_features(df_raw)
        df_clean = drop_low_iv_features(df_clean)
        
        # Faz a previsão
        pred = int(modelo_producao.predict(df_clean)[0])
        prob = float(modelo_producao.predict_proba(df_clean)[0, 1])
        
        # Retorna a resposta como JSON
        return {
            "customerID": cliente.customerID,
            "previsao_churn": pred,
            "probabilidade_churn": round(prob, 4),
            "mensagem": "Alto risco de cancelamento!" if prob > 0.5 else "Cliente estável."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao processar a previsão: {str(e)}")

@app.get("/")
def health_check():
    """Endpoint básico para checar se a API está no ar."""
    return {"status": "A API está Online e pronta para uso!"}