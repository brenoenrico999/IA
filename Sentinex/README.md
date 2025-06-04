# Sentinex - Analisador de Ações com Sentimento de Notícias e Twitter

Este projeto visa prever o comportamento futuro de uma ação com base em dados históricos e em análises de sentimento de notícias e tweets. Utiliza a biblioteca `yfinance` para obter dados históricos da ação, `sklearn` para criar um modelo de previsão, e a API do News e do Twitter para análise de sentimento das notícias e das redes sociais, respectivamente.

## Funcionalidades

- **Previsão de Preço**: Utiliza um modelo de RandomForestRegressor para prever o preço de fechamento da ação no dia seguinte.
- **Análise de Sentimento**:
  - **Notícias**: Obtém notícias relacionadas à ação e calcula o sentimento com base nas manchetes.
  - **Twitter**: Busca tweets recentes sobre a ação e calcula o sentimento com base no texto dos tweets.
- **Conversão de Moeda**: Converte o preço da ação de dólares para reais utilizando taxas de câmbio atuais.

## Requisitos

Antes de executar o projeto, você deve ter o Python instalado em seu sistema. Além disso, você precisará instalar as seguintes bibliotecas:

```bash
pip install yfinance sklearn imbalanced-learn forex-python newsapi-python tweepy textblob
```
Você também precisará das seguintes chaves de API:

Twitter API: Para acessar tweets relacionados à ação.

News API: Para obter notícias financeiras.


## Configuração
Obtenha as Chaves de API:

News API: Crie uma conta em News API e obtenha uma chave de API.

Twitter API: Crie uma conta de desenvolvedor no Twitter Developer e obtenha suas chaves de API e tokens.

Defina as chaves por meio de variáveis de ambiente antes de executar o script:

```bash
export NEWS_API_KEY="sua_chave_newsapi"
export TWITTER_API_KEY="sua_chave_twitter"
export TWITTER_API_SECRET_KEY="seu_secret_key"
export TWITTER_ACCESS_TOKEN="seu_access_token"
export TWITTER_ACCESS_TOKEN_SECRET="seu_access_secret"
```
Caso essas variáveis não sejam definidas, o script executa apenas a previsão de preços
e ignora a análise de sentimento.
## Executando o Código
Para executar o código, simplesmente execute o script Python:

```python
python main.py
```
O script solicitará que você insira o símbolo da ação (por exemplo, AAPL para Apple) e então exibirá a previsão do preço de fechamento para o próximo dia, juntamente com a porcentagem de variação histórica e o sentimento das notícias e tweets relacionados.

## Estrutura do Código
### Obtenção de Dados:

get_stock_data(symbol): Obtém dados históricos da ação usando a API do yfinance.

get_current_price(symbol): Obtém o preço de fechamento atual da ação.
### Preparação dos Dados:

prepare_data(df): Prepara os dados históricos, calculando médias móveis e preparando características para o modelo.

### Modelo de Previsão:

train_model(features, target): Treina um modelo de RandomForestRegressor para prever o preço de fechamento.

predict_price(model, features): Faz a previsão com base nas características fornecidas.
### Análise de Sentimento:

get_news_sentiment(api_key, query): Obtém notícias relacionadas e calcula o sentimento usando a News API.

get_twitter_sentiment(api_key, api_secret_key, access_token, access_token_secret, query): Obtém tweets relacionados e calcula o sentimento usando a API do Twitter.

analyze_sentiment(text): Analisa o sentimento do texto usando a biblioteca TextBlob.
Conversão de Moeda:

get_real_price(usd_price): Converte o preço de USD para BRL usando forex-python.

## Exemplo de Saída
Ao executar o script, você verá uma saída semelhante a:

```ruby
Digite o símbolo da ação (por exemplo, AAPL para Apple): AAPL
Preço de fechamento atual: R$ 175.00
Previsão do preço de fechamento para o próximo dia: R$ 176.50
Porcentagem de subida baseada no histórico: 0.86%
Sentimento das notícias: 0.12
Sentimento do Twitter: 0.05
```

### Licença
Este projeto está licenciado sob a MIT License.