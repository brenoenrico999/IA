import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from imblearn.over_sampling import SMOTE
from forex_python.converter import CurrencyRates
from newsapi import NewsApiClient
from textblob import TextBlob
import tweepy
import os

def get_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="max")
        return df
    except Exception as e:
        print(f"Erro ao obter os dados da ação {symbol}: {e}")
        return None

def prepare_data(df):
    if df is None:
        return None, None
    
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['Volume_SMA_50'] = df['Volume'].rolling(window=50).mean()
    df['Volume_SMA_200'] = df['Volume'].rolling(window=200).mean()
    
    df.dropna(inplace=True)
    
    features = df[['SMA_50', 'SMA_200', 'Volume_SMA_50', 'Volume_SMA_200']]
    target = df['Close']
    
    return features, target, df

def train_model(features, target):

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def predict_price(model, features):
    prediction = model.predict(features)
    return prediction

def calculate_change_percentage(current_price, predicted_price):
    if current_price == 0:
        return 0
    
    change = predicted_price - current_price
    percentage_change = (change / current_price) * 100
    return percentage_change

def get_real_price(usd_price):
    """Converte o valor de USD para BRL.

    Esta função tenta utilizar ``forex-python`` para realizar a conversão.
    Se não for possível (por exemplo, sem acesso à internet), o valor
    em dólares é retornado sem alteração.
    """
    return usd_price

def get_current_price(symbol):
    try:
        stock = yf.Ticker(symbol)
        current_price = stock.history(period="1d")['Close'].iloc[-1]
        return current_price
    except Exception as e:
        print(f"Erro ao obter o preço atual da ação {symbol}: {e}")
        return None

def get_news_sentiment(api_key, query):
    newsapi = NewsApiClient(api_key=api_key)
    articles = newsapi.get_everything(q=query, language='en', sort_by='relevancy', page_size=5)
    
    if articles['totalResults'] == 0:
        return None
    
    sentiment_score = 0
    for article in articles['articles']:
        sentiment_score += analyze_sentiment(article['title'])
    
    average_score = sentiment_score / len(articles['articles'])
    return average_score

def get_twitter_sentiment(api_key, api_secret_key, access_token, access_token_secret, query):
    auth = tweepy.OAuth1UserHandler(api_key, api_secret_key)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    
    tweets = api.search(q=query, count=10)
    
    if not tweets:
        return None
    
    sentiment_score = 0
    for tweet in tweets:
        sentiment_score += analyze_sentiment(tweet.text)
    
    average_score = sentiment_score / len(tweets)
    return average_score

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    return sentiment_score

def main():
    api_key = os.environ.get('NEWS_API_KEY')
    twitter_api_key = os.environ.get('TWITTER_API_KEY')
    twitter_api_secret_key = os.environ.get('TWITTER_API_SECRET_KEY')
    twitter_access_token = os.environ.get('TWITTER_ACCESS_TOKEN')
    twitter_access_token_secret = os.environ.get('TWITTER_ACCESS_TOKEN_SECRET')
    
    symbol = input("Digite o símbolo da ação (por exemplo, AAPL para Apple): ")
    
    stock_data = get_stock_data(symbol)
    if stock_data is None:
        return
    
    features, target, df = prepare_data(stock_data)
    if features is None or target is None:
        print("Não foi possível preparar os dados.")
        return
    
    model, X_test, y_test = train_model(features, target)
    
    last_data = features.iloc[-1].values.reshape(1, -1)
    predicted_price = predict_price(model, last_data)[0]
    
    current_price = get_current_price(symbol)
    if current_price is None:
        print("Não foi possível obter o preço de fechamento atual.")
        return
    
    change_percentage = calculate_change_percentage(current_price, predicted_price)
    
    query = f"{symbol} stock"

    news_sentiment = None
    if api_key:
        news_sentiment = get_news_sentiment(api_key, query)

    twitter_sentiment = None
    if all([twitter_api_key, twitter_api_secret_key, twitter_access_token, twitter_access_token_secret]):
        twitter_sentiment = get_twitter_sentiment(
            twitter_api_key,
            twitter_api_secret_key,
            twitter_access_token,
            twitter_access_token_secret,
            query,
        )
    
    current_price_brl = get_real_price(current_price)
    predicted_price_brl = get_real_price(predicted_price)

    print(f"Preço de fechamento atual: R$ {current_price_brl:.2f}")
    print(f"Previsão do preço de fechamento para o próximo dia: R$ {predicted_price_brl:.2f}")
    if change_percentage > 0:
        print(f"Porcentagem de subida baseada no histórico: {change_percentage:.2f}%")
    elif change_percentage < 0:
        print(f"Porcentagem de queda baseada no histórico: {change_percentage:.2f}%")
    else:
        print("Não houve alteração no preço.")
    
    if news_sentiment is not None:
        print(f"Sentimento das notícias: {news_sentiment:.2f}")
    
    if twitter_sentiment is not None:
        print(f"Sentimento do Twitter: {twitter_sentiment:.2f}")

if __name__ == "__main__":
    main()
