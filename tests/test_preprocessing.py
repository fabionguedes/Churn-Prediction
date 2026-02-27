import pytest
import pandas as pd
import numpy as np
import sys
import os

# TRUQUE DE MESTRE: Garantir que a pasta raiz do projeto seja reconhecida
# para que possamos importar os arquivos da pasta 'src' sem erros
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importando as funções que queremos testar
from src.a3data.preprocessing import clean_numeric_features, encode_target

def test_clean_numeric_features():
    """
    Testa se as vírgulas estão sendo substituídas por pontos, 
    se as colunas viram números (float) e se a regra de nulos funciona.
    """
    # 1. Cria um DataFrame falso (Mock) para testar os cenários críticos
    mock_data = pd.DataFrame({
        'MonthlyCharges': ['50,50', 20.0, '30,00'],
        'TotalCharges': ['100,50', np.nan, ' '] # O segundo cliente tem TotalCharges nulo (1º mês)
    })
    
    # 2. Aplica a sua função de produção
    df_clean = clean_numeric_features(mock_data)
    
    # 3. Asserts (Garantias)
    # Garante que '50,50' virou o número 50.50
    assert df_clean['MonthlyCharges'].iloc[0] == 50.50 
    
    # Garante a regra de negócio: Se TotalCharges é nulo, copia o MonthlyCharges (20.0)
    assert df_clean['TotalCharges'].iloc[1] == 20.0
    
    # Garante que as colunas realmente viraram numéricas no Pandas
    assert pd.api.types.is_numeric_dtype(df_clean['MonthlyCharges'])
    assert pd.api.types.is_numeric_dtype(df_clean['TotalCharges'])

def test_encode_target():
    """
    Testa se a variável 'Churn' é corretamente mapeada de Yes/No para 1/0.
    """
    # 1. Cria DataFrame falso
    mock_data = pd.DataFrame({
        'Churn': ['Yes', 'No', 'Yes', 'No']
    })
    
    # 2. Aplica a função
    df_encoded = encode_target(mock_data)
    
    # 3. Asserts
    assert df_encoded['Churn'].iloc[0] == 1
    assert df_encoded['Churn'].iloc[1] == 0
    assert df_encoded['Churn'].sum() == 2 # Deve haver exatamente dois números '1'