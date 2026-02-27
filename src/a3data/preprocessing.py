import os
import pandas as pd
import numpy as np

def load_data(file_path: str) -> pd.DataFrame:
    """Carrega os dados brutos de um arquivo CSV."""
    # Adicionamos um tratamento de erro básico para produção
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado no caminho: {file_path}")
    return pd.read_csv(file_path)

def clean_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Substitui vírgulas por pontos e converte colunas de cobrança para float.
    Aplica regra de negócio: Clientes no 1º mês (TotalCharges nulo) recebem o valor de MonthlyCharges.
    """
    df = df.copy() # Boa prática: evitar SettingWithCopyWarning
    
    for col in ['MonthlyCharges', 'TotalCharges']:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(str).str.replace(',', '.')
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Preenchimento de dados faltantes (Regra de Negócio)
    df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'])
    
    return df

def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Converte a variável alvo 'Churn' para formato binário (1 e 0)."""
    df = df.copy()
    if 'Churn' in df.columns:
        df['Churn'] = np.where(df['Churn'] == 'Yes', 1, 0)
    return df

def drop_low_iv_features(df: pd.DataFrame) -> pd.DataFrame:
    """Remove features que não agregam valor preditivo baseado na análise de Information Value (IV)."""
    cols_to_drop = ['MultipleLines', 'SeniorCitizen', 'customerID', 'gender', 'PhoneService']
    
    # Interseção garante que não haverá erro se a coluna já tiver sido removida
    cols_present = [col for col in cols_to_drop if col in df.columns]
    return df.drop(columns=cols_present)

def run_preprocessing_pipeline(input_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Função principal que orquestra todas as etapas de limpeza.
    """
    print("Iniciando pré-processamento...")
    df = load_data(input_path)
    
    df = clean_numeric_features(df)
    df = encode_target(df)
    df = drop_low_iv_features(df)
    
    if output_path:
        # Cria o diretório de saída se não existir
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, sep=';', decimal='.')
        print(f"Dados processados salvos com sucesso em: {output_path}")
        
    return df

# --- TESTE DO PIPELINE ---
if __name__ == "__main__":
    # Em produção, esses caminhos viriam de variáveis de ambiente (.env) ou de um arquivo config.py
    RAW_DATA_PATH = 'data/raw/Customer-Churn - Customer-Churn.csv'
    CLEAN_DATA_PATH = 'data/curated/Customer-Churn_iv_removed.csv'
    
    # Você pode testar localmente executando este script
    df_clean = run_preprocessing_pipeline(RAW_DATA_PATH, CLEAN_DATA_PATH)