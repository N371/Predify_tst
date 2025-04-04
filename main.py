from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from io import StringIO
import pandas as pd
import ollama
import os
import logging
import traceback

# Configuração
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Caminhos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'sales.csv')
OLLAMA_MODEL = "phi3"

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def install_missing_dependencies():
    try:
        import tabulate
    except ImportError:
        logger.info("Instalando dependência tabulate...")
        import subprocess
        subprocess.run(["pip", "install", "tabulate"], check=True)

def get_ollama_response(prompt: str) -> str:
    try:
        response = ollama.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            options={'temperature': 0}
        )
        return response['response']
    except Exception as e:
        logger.error(f"Erro Ollama: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(500, "Erro no servidor de modelos")

@app.post("/upload")
async def upload_file(file: UploadFile):
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))

        required_columns = {'date', 'product', 'quantity', 'total'}
        if missing := required_columns - set(df.columns):
            raise ValueError(f"Colunas faltando: {missing}")

        df.to_csv(CSV_PATH, index=False)
        return {"message": "Arquivo salvo com sucesso", "columns": list(df.columns)}
    except Exception as e:
        logger.error(f"Upload error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(400, detail=str(e))

@app.post("/ask")
async def ask(question: str = Form(...)):
    install_missing_dependencies()

    if not os.path.exists(CSV_PATH):
        raise HTTPException(404, "Arquivo não encontrado. Faça upload primeiro.")

    try:
        df = pd.read_csv(CSV_PATH)
        try:
            data_preview = df.head().to_markdown()
        except:
            data_preview = df.head().to_string()

        context = f"""
        Analise estes dados de vendas:
        {data_preview}

        Pergunta: {question}
        Responda com valores exatos quando possível.
        """
        answer = get_ollama_response(context)
        return {"question": question, "answer": answer}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(500, "Erro ao processar a pergunta")

@app.get("/health")
async def health_check():
    try:
        status = {
            "csv_loaded": os.path.exists(CSV_PATH),
            "ollama_running": False,
            "model_available": False
        }

        try:
            models = ollama.list()['models']
            status.update({
                "ollama_running": True,
                "model_available": any(model['name'] == OLLAMA_MODEL for model in models)
            })
        except:
            pass

        return status
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        raise HTTPException(500, "Erro na verificação de saúde")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
