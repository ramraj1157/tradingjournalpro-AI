from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from app.routes import sentiment 
from app.routes import technical 

app = FastAPI(default_response_class=ORJSONResponse)

# ✅ Allow CORS for all origins (fix Render CORS issue)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔥 Change this if you want to restrict specific origins
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
