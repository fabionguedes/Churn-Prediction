import argparse
import uvicorn
import os

# Ajustamos o caminho adicionando o ".a3data" para o Python encontrar os arquivos
from src.a3data.preprocessing import run_preprocessing_pipeline
from src.a3data.train import train_and_evaluate

def main():
    parser = argparse.ArgumentParser(
        description="Maestro do Projeto de Previsão de Churn",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--step', 
        type=str, 
        required=True, 
        choices=['preprocess', 'train', 'api'],
        help="Escolha a etapa do pipeline a ser executada:\n"
             "  preprocess : Limpa os dados brutos e gera a base curada.\n"
             "  train      : Treina o modelo XGBoost e salva o arquivo .pkl.\n"
             "  api        : Inicia o servidor Web (FastAPI)."
    )

    args = parser.parse_args()

    # Caminhos dos arquivos
    RAW_DATA_PATH = '/Users/guedesf/Documents/Data Science/Projetos/a3data/data/raw/Customer-Churn - Customer-Churn.csv'
    CLEAN_DATA_PATH = '/Users/guedesf/Documents/Data Science/Projetos/a3data/data/curated/Customer-Churn_iv_removed.csv'
    MODEL_SAVE_PATH = '/Users/guedesf/Documents/Data Science/Projetos/a3data/models/champion_xgb_pipeline.pkl'

    # --- Roteador de Execução ---
    if args.step == 'preprocess':
        print("\n>>> [ETAPA 1] Iniciando o Pré-processamento dos Dados...")
        run_preprocessing_pipeline(RAW_DATA_PATH, CLEAN_DATA_PATH)
        
    elif args.step == 'train':
        print("\n>>> [ETAPA 2] Iniciando o Treinamento do Modelo Campeão...")
        train_and_evaluate(CLEAN_DATA_PATH, MODEL_SAVE_PATH)
        
    elif args.step == 'api':
        print("\n>>> [ETAPA 3] Iniciando o Servidor de API (FastAPI)...")
        # Ajustamos o caminho aqui também adicionando ".a3data"
        uvicorn.run("src.a3data.api:app", host="127.0.0.1", port=8080, reload=True)

if __name__ == "__main__":
    main()