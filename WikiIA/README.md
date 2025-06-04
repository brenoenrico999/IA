# WikiIA

Esta é uma aplicação que coleta dados de páginas da Wikipedia, permite fazer perguntas sobre os dados coletados usando um modelo de perguntas e respostas baseado em BERT (Bidirectional Encoder Representations from Transformers), e mantém o controle dos tópicos já coletados.

## Requisitos
- Python 3.x
- Bibliotecas: `wikipedia-api`, `beautifulsoup4`, `transformers`, `torch`, `datetime`, `json`

## Instalação
1. Clone este repositório para o seu computador.
2. Certifique-se de que o Python 3.x esteja instalado. Caso não esteja instalado, baixe-o em https://www.python.org/downloads/ e siga as instruções de instalação para o seu sistema operacional.
3. Instale as bibliotecas necessárias executando o seguinte comando no terminal ou prompt de comando:
```
pip install wikipedia-api beautifulsoup4 transformers torch
```
## Como usar

1. Navegue para o diretório raiz do projeto onde se encontra o arquivo `WikiIA.py`.
2. Execute o seguinte comando para iniciar a aplicação:
```
python WikiIA.py
```
3. O programa irá iniciar a coleta de dados das páginas da Wikipedia definidas no código. Após a coleta, você poderá fazer perguntas sobre os dados coletados.

## Funcionalidades

- Coleta de Dados: A aplicação coleta dados das páginas da Wikipedia definidas no código. Os dados são pré-processados e armazenados em um banco de dados SQLite.

- Perguntas e Respostas: A aplicação permite fazer perguntas sobre os dados coletados. Ela utiliza um modelo de perguntas e respostas BERT para encontrar as respostas.

- Atualização do Banco de Dados: A aplicação permite atualizar o banco de dados, apagando as tabelas existentes e recriando-as. Isso é útil para limpar os dados antigos e coletar novos dados.

## Observações

- Certifique-se de que o modelo BERT (neuralmind/bert-base-portuguese-cased) está instalado na pasta trained_model antes de executar o programa.

- Os dados coletados são armazenados no arquivo `wiki_data.db`, que é criado automaticamente.

- Durante a coleta, os dados são pré-processados para remover tags HTML, números entre colchetes e espaços em excesso.

- É possível adicionar novos tópicos para coleta durante a execução do programa.

- A aplicação mantém o controle dos tópicos já coletados e permite verificar o status das coletas.

- O modelo de perguntas e respostas BERT é treinado com um conjunto de perguntas e respostas pré-definidas, é possível alterar esse modelo ou treinar um próprio.

## Licença
Este projeto está licenciado sob a licença MIT.


