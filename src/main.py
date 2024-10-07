from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from src.utils.routes import user_routes, admin_routes
import warnings
from urllib3.exceptions import InsecureRequestWarning

# Ignorer les avertissements liés aux certificats
warnings.simplefilter('ignore', InsecureRequestWarning)

app = FastAPI(
    title="Evaluation Avis Clients",
    description="Cette API permet de prédire les avis clients",
    version="2.0.0",
    openapi_tags=[
        {'name': 'user', 'description': 'Endpoints pour les utilisateurs'},
        {'name': 'admin', 'description': 'Endpoints réservés à l\'administrateur de l\'API'}
    ]
)

# Configuration de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Changez "*" pour une liste spécifique de domaines si nécessaire
    allow_credentials=True,
    allow_methods=["*"],  # Vous pouvez restreindre cela à une liste spécifique de méthodes comme ["GET", "POST"]
    allow_headers=["*"],  # Vous pouvez restreindre cela à une liste spécifique d'en-têtes
)

# Inclusion des routes utilisateurs et admin
app.include_router(user_routes)
app.include_router(admin_routes)

# Servir les fichiers statiques (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="src/static"), name="static")

# Routes pour les pages HTML
@app.get("/", response_class=FileResponse)
async def root():
    return FileResponse("src/static/index.html")

@app.get("/register", response_class=FileResponse)
async def register():
    return FileResponse("src/static/register.html")

@app.get("/main", response_class=FileResponse)
async def main():
    return FileResponse("src/static/main.html")

@app.get("/manage_access", response_class=FileResponse)
async def manage_access():
    return FileResponse("src/static/manage_access.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
