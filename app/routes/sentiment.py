from fastapi import APIRouter
from ..services.sentiment_service import get_stock_sentiment

router = APIRouter()

@router.get("/{stock_name}")  # ✅ This defines the route
async def analyze_stock_sentiment(stock_name: str):
    """Fetch news & analyze sentiment for an Indian stock"""
    sentiment_data = get_stock_sentiment(stock_name)

    if not sentiment_data:
        return {"error": "No news found for this stock."}

    return {"stock": stock_name, "analysis": sentiment_data}
