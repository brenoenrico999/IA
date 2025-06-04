Este arquivo contém instruções básicas para agentes automatizados que fazem modificações neste repositório.

## Guia de Contribuição

1. **Estrutura do repositório**:
   - `BitcoinRNN/` : Código para previsão de preços do Bitcoin usando redes neurais.
   - `Sentinex/`   : Projeto de análise de ações com base em notícias e Twitter.
   - `WikiIA/`     : Coletor de dados da Wikipédia e sistema de perguntas e respostas.

2. **Estilo de código**:
   - Utilize convenções PEP8 para arquivos Python.
   - Prefira comentários em português quando aplicável.

3. **Verificações recomendadas**:
   - Execute `python -m py_compile $(git ls-files '*.py')` antes de cada commit para garantir que todos os scripts estão com a sintaxe correta.

4. **Commits**:
   - Escreva mensagens de commit concisas em português.
   - Não modifique commits antigos (sem uso de amend ou rebase).

5. **Pull Requests**:
   - Descreva brevemente as alterações e indique como testá-las.

## Execução dos Projetos

1. Utilize **Python 3.8** ou superior. É recomendável criar um ambiente virtual:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Instale as dependências de cada módulo:

   - **BitcoinRNN**
     ```bash
     pip install numpy pandas scikit-learn tensorflow matplotlib
     ```
     Para rodar o exemplo principal:
     ```bash
     python BitcoinRNN/main.py
     ```

   - **Sentinex**
     ```bash
     pip install yfinance sklearn imbalanced-learn forex-python newsapi-python tweepy textblob
     ```
     Algumas partes exigem chaves de API externas.
     ```bash
     python Sentinex/main.py
     ```

   - **WikiIA**
     ```bash
     pip install wikipedia-api beautifulsoup4 transformers sqlite3
     ```
     Certifique-se de que o modelo BERT esteja em `WikiIA/trained_model/`.
     ```bash
     python WikiIA/WikiIA.py
     ```

3. Scripts auxiliares como `WikiIA/treino.py` podem requerer `torch` e são executados de maneira similar:
   ```bash
   pip install torch
   python WikiIA/treino.py
   ```

4. Consulte sempre os READMEs de cada pasta para detalhes complementares e exemplos de uso.


## Testes

1. **BitcoinRNN**
   - Dependencias: `numpy`, `pandas`, `scikit-learn`, `tensorflow`, `matplotlib`.
   - Teste rapido:
     ```bash
     python -m py_compile BitcoinRNN/*.py
     python BitcoinRNN/main.py
     ```
     Use um arquivo `bitcoin_price_data.csv` com os dados de exemplo.

2. **Sentinex**
   - Dependencias: `yfinance`, `sklearn`, `imbalanced-learn`, `forex-python`, `newsapi-python`, `tweepy`, `textblob`.
   - Defina as chaves `NEWS_API_KEY`, `TWITTER_API_KEY`, `TWITTER_API_SECRET_KEY`, `TWITTER_ACCESS_TOKEN` e `TWITTER_ACCESS_TOKEN_SECRET` antes de executar.
   - Teste rapido:
     ```bash
     python -m py_compile Sentinex/*.py
     NEWS_API_KEY=... TWITTER_API_KEY=... python Sentinex/main.py
     ```

3. **WikiIA**
   - Dependencias: `wikipedia-api`, `beautifulsoup4`, `transformers`, `sqlite3`. O script de treino requer `torch`.
   - Verifique que o modelo BERT esta em `WikiIA/trained_model/`.
   - Teste rapido:
     ```bash
     python -m py_compile WikiIA/*.py
     python WikiIA/WikiIA.py
     ```
   - Para treinar novamente:
     ```bash
     pip install torch
     python WikiIA/treino.py
     ```

Estes testes basicos validam cada modulo individualmente. Consulte os READMEs para configuracoes mais detalhadas.
