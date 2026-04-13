import os
import io
import asyncio
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes
from PIL import Image
from google import genai
from dotenv import load_dotenv

# CONFIG
load_dotenv()

class AppConfig:
    POPPLER_PATH = os.getenv("POPPLER_PATH", None)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEYS")
    MODEL_NAME = os.getenv("MODEL", "gemini-1.0-pro-vision")
    MAX_PDF_PAGES = 8
    IMAGE_SIZE = (1200, 1200)


if not AppConfig.GEMINI_API_KEY:
    raise ValueError("Falta GEMINI_API_KEY")

client = genai.Client(api_key=AppConfig.GEMINI_API_KEY)

# MODELOS
class ItemFactura(BaseModel):
    descripcion: str
    cantidad: Optional[float] = None
    precio_unit: Optional[float] = None
    total: Optional[float] = None

class DatosFactura(BaseModel):
    proveedor: Optional[str] = None
    nit: Optional[str] = None
    numero_factura: Optional[str] = None
    fecha: Optional[str] = None
    subtotal: Optional[float] = None
    iva: Optional[float] = None
    total: Optional[float] = None
    items: List[ItemFactura] = Field(default_factory=list)

# PROMPT
PROMPT = """
Extract invoice data from the document.

The document may contain multiple pages of the same invoice.

Rules:
- Do not infer missing values
- Merge all pages into one invoice
- Do not duplicate items
- Merge multi-line descriptions into one
- Include only real line items (products/services)
- Ignore headers, footers, and repeated text
"""

# IA
def invocar_ia(imagenes: list) -> DatosFactura:
    try:
        response = client.models.generate_content(
            model=AppConfig.MODEL_NAME,
            contents=[PROMPT] + imagenes,
            config={
                "response_mime_type": "application/json",
                "response_schema": DatosFactura,
                "temperature": 0.1
            }
        )
        return DatosFactura.model_validate_json(response.text)

    except Exception as e:
        raise RuntimeError(f"Error IA: {str(e)}")

# PROCESAMIENTO
def procesar_archivo(file_bytes: bytes, filename: str) -> dict:
    try:
        ext = filename.lower().split('.')[-1]
        imagenes = []

        if ext == "pdf":
            paginas = convert_from_bytes(
                file_bytes,
                poppler_path=AppConfig.POPPLER_PATH,
                size=AppConfig.IMAGE_SIZE
            )

            if len(paginas) > AppConfig.MAX_PDF_PAGES:
                return {"estado": "ERROR_TAMANIO", "error": "PDF demasiado grande"}

            imagenes = paginas

        elif ext in ["jpg", "jpeg", "png"]:
            img = Image.open(io.BytesIO(file_bytes))
            img.thumbnail(AppConfig.IMAGE_SIZE)
            imagenes = [img]

        else:
            return {"estado": "ERROR_FORMATO", "error": "Formato no soportado"}

        datos = invocar_ia(imagenes)

        return {
            "estado": "PROCESADA",
            "datos": datos.model_dump()
        }

    except ValueError as e:
        return {"estado": "ERROR_DATOS", "error": str(e)}

    except Exception as e:
        return {"estado": "ERROR_PROCESAMIENTO", "error": str(e)}

# API
app = FastAPI(title="API Facturas IA", version="1.0")

# CORS: permitir llamadas desde tu frontend (ajusta los orígenes en producción)
app.add_middleware(
    CORSMiddleware,
    # Dominio público de tu frontend en InfinityFree
    allow_origins=["https://facturasmart.page.gd"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/procesar")
async def procesar_facturas(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron archivos")

    async def procesar(file: UploadFile):
        contenido = await file.read()
        resultado = await asyncio.to_thread(procesar_archivo, contenido, file.filename)
        return {
            "archivo": file.filename,
            "resultado": resultado
        }

    tareas = [procesar(f) for f in files]
    resultados = await asyncio.gather(*tareas)

    return JSONResponse(
        content={
            "total": len(resultados),
            "facturas": resultados
        },
        status_code=status.HTTP_200_OK
    )

# MODO LOCAL
if __name__ == "__main__":
    import json
    import time

    carpeta = "facturas"
    salida = "resultados.json"
    resultados = []

    print("---- MODO LOCAL ----")

    if os.path.exists(carpeta):
        for archivo in os.listdir(carpeta):
            if archivo.lower().endswith((".jpg", ".jpeg", ".png", ".pdf")):
                print(f"Procesando: {archivo}")

                with open(os.path.join(carpeta, archivo), "rb") as f:
                    contenido = f.read()

                res = procesar_archivo(contenido, archivo)
                resultados.append({"archivo": archivo, "resultado": res})

                time.sleep(1)  # evitar rate limit

        with open(salida, "w", encoding="utf-8") as f:
            json.dump(resultados, f, indent=4, ensure_ascii=False)

        print(f"\nResultados guardados en {salida}")
    else:
        print("Carpeta 'facturas' no encontrada")
