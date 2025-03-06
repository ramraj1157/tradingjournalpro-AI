import requests
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load FinBERT Model
tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

def fetch_indian_stock_news(stock_name: str):
    """Fetch latest news from MoneyControl for an Indian stock"""
    search_url = f"https://www.moneycontrol.com/news/tags/{stock_name}.html"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(search_url, headers=headers)
    if response.status_code != 200:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    articles = soup.find_all("li", class_="clearfix")[:5]  # Get top 5 headlines

    news_list = []
    for a in articles:
        title_tag = a.find("a")
        if title_tag:
            title = title_tag.text.strip()
            link = title_tag["href"]
            news_list.append({"headline": title, "link": link})
    
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
    news = fetch_indian_stock_news(stock_name)
    results = []

    for item in news:
        sentiment_result = analyze_sentiment(item["headline"])
        results.append({**item, **sentiment_result})
    
    return results
