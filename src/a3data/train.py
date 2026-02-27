import os
import pandas as pd
import numpy as np
import joblib # Biblioteca para salvar o modelo treinado
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import recall_score, classification_report
from xgboost import XGBClassifier

def load_clean_data(file_path: str) -> pd.DataFrame:
    """Carrega os dados limpos da Fase 1."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}. Rode o preprocessing.py primeiro.")
    return pd.read_csv(file_path, sep=';')

def calculate_scale_pos_weight(y_train: pd.Series) -> float:
    """Calcula o peso para classes desbalanceadas baseado no conjunto de treino."""
    count_negative = (y_train == 0).sum()
    count_positive = (y_train == 1).sum()
    return count_negative / count_positive

def build_model_pipeline(categorical_cols: list, scale_pos_weight: float, best_params: dict) -> Pipeline:
    """
    Constrói o pipeline completo contendo o One-Hot Encoding e o modelo XGBoost.
    """
    # 1. Feature Engineering
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough' # Mantém as numéricas intactas (XGBoost lida bem com elas sem Scaling)
    )

    # 2. Algoritmo Campeão com os hiperparâmetros fixos
    xgb_model = XGBClassifier(
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        eval_metric='logloss',
        **best_params # Desempacota o dicionário de hiperparâmetros
    )

    # 3. Pipeline Final
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb_model)
    ])
    
    return pipeline

def train_and_evaluate(data_path: str, model_save_path: str):
    """
    Orquestra o carregamento, separação, treinamento, avaliação e salvamento do modelo.
    """
    print("Iniciando o processo de treinamento...")
    
    # 1. Carregar dados limpos
    df = load_clean_data(data_path)
    
    # 2. Separar X e y
    X = df.drop(columns=['Churn'])
    y = df['Churn'].astype(int)
    
    # 3. Divisão Treino e Teste (Estratificada)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Calcular pesos e identificar colunas
    scale_weight = calculate_scale_pos_weight(y_train)
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Treinando XGBoost com scale_pos_weight = {scale_weight:.4f}...")
    
    # ---------------------------------------------------------
    # INSIRA AQUI OS MELHORES PARÂMETROS ENCONTRADOS NO NOTEBOOK
    # ---------------------------------------------------------
    BEST_PARAMS = {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    
    # 5. Construir e Treinar o Pipeline
    pipeline = build_model_pipeline(categorical_cols, scale_weight, BEST_PARAMS)
    pipeline.fit(X_train, y_train)
    
    # 6. Avaliar no conjunto de Teste
    print("\n--- Avaliação no Conjunto de Teste ---")
    y_pred = pipeline.predict(X_test)
    recall = recall_score(y_test, y_pred)
    
    print(f"Recall no Teste: {recall:.4f}")
    print("\nRelatório de Classificação Completo:")
    print(classification_report(y_test, y_pred))
    
    # 7. Salvar o modelo em disco
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(pipeline, model_save_path)
    print(f"\nModelo treinado e salvo com sucesso em: {model_save_path} 🚀")


# --- EXECUÇÃO DO SCRIPT ---
if __name__ == "__main__":
    # Ajuste os caminhos absolutos conforme a sua máquina, se necessário
    CLEAN_DATA_PATH = '/Users/guedesf/Documents/Data Science/Projetos/a3data/data/curated/Customer-Churn_iv_removed.csv'
    MODEL_SAVE_PATH = '/Users/guedesf/Documents/Data Science/Projetos/a3data/models/champion_xgb_pipeline.pkl'
    
    train_and_evaluate(CLEAN_DATA_PATH, MODEL_SAVE_PATH)