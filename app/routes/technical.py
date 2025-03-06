from fastapi import APIRouter, HTTPException
from ..services.technical_analysis import fetch_stock_data, calculate_technical_indicators, prepare_ml_data, train_ml_model, predict_next_day


router = APIRouter()

@router.get("/{stock_name}")
async def get_technical_analysis(stock_name: str):
    df = fetch_stock_data(stock_name)
    
    if df is None:
        raise HTTPException(status_code=404, detail="Stock data not found")

    df = calculate_technical_indicators(df)
    X, y = prepare_ml_data(df)

    if X is None or y is None:
        raise HTTPException(status_code=400, detail="Insufficient data for analysis")

    model = train_ml_model(X, y)
    prediction = predict_next_day(model, X[-1], df)

    return {"stock": stock_name, "prediction": prediction}
