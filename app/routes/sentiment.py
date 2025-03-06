from fastapi import APIRouter, HTTPException
from app.services.sentiment_service import get_stock_sentiment

router = APIRouter()

@router.get("/{stock_name}")
async def fetch_sentiment(stock_name: str):
    """API Endpoint to get sentiment analysis for a stock"""
    result = get_stock_sentiment(stock_name)
    # if not result:
    #     raise HTTPException(status_code=404, detail="No news articles found or API limit reached.")
    return {"stock": stock_name, "analysis": result}
