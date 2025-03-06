from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from app.routes import sentiment 
from app.routes import technical 

app = FastAPI(default_response_class=ORJSONResponse)

# ✅ CORS Config (No empty values)
origins = [
    "https://tradingjournalpro.vercel.app",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Register Routes
app.include_router(sentiment.router, prefix="/sentiment", tags=["Sentiment Analysis"])
app.include_router(technical.router, prefix="/technical", tags=["Technical Analysis"])

@app.get("/")
def root():
    return {"message": "Welcome to the Trading Analysis API"}
