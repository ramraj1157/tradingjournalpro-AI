import requests
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# API Key for NewsAPI
NEWS_API_KEY = "1bb2003b921d45619da52415e629215c"

# Load FinBERT Model
tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

def fetch_stock_news(stock_name: str):
    """Fetch latest stock news from NewsAPI"""
    url = f"https://newsapi.org/v2/everything?q={stock_name}&apiKey={NEWS_API_KEY}&language=en"
    
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching news: {response.status_code}")
        return []
    
    data = response.json()
    articles = data.get("articles", [])[:5]  # Get top 5 news articles

    news_list = [
        {"headline": article["title"], "link": article["url"]}
        for article in articles
        if article["title"] and article["url"]
    ]
    
    return news_list

def analyze_sentiment(text: str):
    """Analyze sentiment using FinBERT"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=1).tolist()[0]

    labels = ["Negative", "Neutral", "Positive"]
    sentiment = labels[scores.index(max(scores))]

    return {"sentiment": sentiment, "scores": {"Negative": scores[0], "Neutral": scores[1], "Positive": scores[2]}}

def get_stock_sentiment(stock_name: str):
    """Fetch news & analyze sentiment"""
    news_articles = fetch_stock_news(stock_name)
    results = []

    for article in news_articles:
        sentiment_result = analyze_sentiment(article["headline"])
        results.append({**article, **sentiment_result})
    
    return results

# Example Usage
stock_name = "Reliance"
stock_sentiment = get_stock_sentiment(stock_name)
for news in stock_sentiment:
    print(news)
