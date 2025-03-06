from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import sentiment  # ✅ Import sentiment router
from app.routes import technical

app = FastAPI(title="Trading Analysis API")

origins = [
    "http://localhost.tiangolo.com",
    "https://tradingjournalpro.vercel.app",
    "http://localhost",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ✅ Register the sentiment router
app.include_router(sentiment.router, prefix="/sentiment", tags=["Sentiment Analysis"])
app.include_router(technical.router, prefix ="/technical", tags=["Technical Analysis"])


@app.get("/")
def root():
    return {"message": "Welcome to the Trading Analysis API"}
