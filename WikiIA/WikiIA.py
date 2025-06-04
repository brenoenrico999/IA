import os
import wikipediaapi
import re
from bs4 import BeautifulSoup
import threading
from transformers import pipeline, AutoTokenizer
import sqlite3
from datetime import datetime, timedelta
import json
from transformers import BertTokenizer, BertForQuestionAnswering
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW
import torch

# Carregar o tokenizer
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

# Verificar se o banco de dados já existe
db_exists = os.path.isfile("wiki_data.db")

# Criar o banco de dados e as tabelas se não existirem
conn = sqlite3.connect("wiki_data.db")
if not db_exists:
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS dados (id INTEGER PRIMARY KEY AUTOINCREMENT, pagina TEXT, texto TEXT, data TEXT)")
    cursor.execute("CREATE TABLE IF NOT EXISTS perguntas_respostas (id INTEGER PRIMARY KEY AUTOINCREMENT, pergunta TEXT, resposta TEXT)")
    conn.commit()

class WikipediaCollector(threading.Thread):
    def __init__(self, page):
        threading.Thread.__init__(self)
        self.page = page

    def run(self):
        print("Coletando dados da página:", self.page)
        # Coleta de dados da Wikipedia
        try:
            wiki_wiki = wikipediaapi.Wikipedia(
                language='pt',
                extract_format=wikipediaapi.ExtractFormat.HTML,
                user_agent='beyond'
            )
            page_py = wiki_wiki.page(self.page)
            if page_py.exists():
                content = page_py.text

                # Pré-processamento dos dados
                print("Passo 1: Removendo tags HTML...")
                soup = BeautifulSoup(content, 'html.parser')
                cleaned_text = soup.get_text()

                print("Passo 2: Removendo números entre colchetes...")
                cleaned_text = re.sub(r'\[[0-9]+\]', '', cleaned_text)

                print("Passo 3: Removendo espaços em excesso...")
                cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

                # Raspagem de dados estruturados
                print("Passo 4: Raspagem de dados estruturados...")
                structured_data = {}
                infobox = soup.find("table", {"class": "infobox"})
                if infobox:
                    rows = infobox.find_all("tr")
                    for row in rows:
                        header = row.find("th")
                        if header:
                            key = header.get_text().strip()
                            value = row.find("td").get_text().strip()
                            structured_data[key] = value

                # Tokenização dos dados
                print("Passo 5: Tokenizando os dados...")
                encoded_inputs = tokenizer(cleaned_text, truncation=True, padding=True, max_length=512)

                # Obter os tokens e os rótulos de atenção
                tokens = encoded_inputs["input_ids"]
                attention_mask = encoded_inputs["attention_mask"]

                # Converta os tokens para uma string separada por espaços
                tokens_str = " ".join(tokenizer.convert_ids_to_tokens(tokens))

                # Armazenar os dados no banco de dados
                conn_thread = sqlite3.connect("wiki_data.db")
                cursor = conn_thread.cursor()
                cursor.execute("INSERT INTO dados (pagina, texto, data) VALUES (?, ?, ?)",
                               (self.page, json.dumps({"text": cleaned_text, "structured_data": structured_data}),
                                datetime.now().strftime("%Y-%m-%d")))
                conn_thread.commit()
                conn_thread.close()

        except Exception as e:
            print("Erro ao coletar dados da página:", self.page)
            print("Mensagem de erro:", e)

def atualizar_banco():
    print("Atualizando banco de dados...")
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS dados")
    cursor.execute("DROP TABLE IF EXISTS perguntas_respostas")
    cursor.execute("CREATE TABLE IF NOT EXISTS dados (id INTEGER PRIMARY KEY AUTOINCREMENT, pagina TEXT, texto TEXT, data TEXT)")
    cursor.execute("CREATE TABLE IF NOT EXISTS perguntas_respostas (id INTEGER PRIMARY KEY AUTOINCREMENT, pergunta TEXT, resposta TEXT)")
    conn.commit()

# Lista de páginas da Wikipedia a serem coletadas
paginas = ["Inteligência artificial", "Aprendizado de máquina", "Processamento de linguagem natural"]
coletas = []

def listar_topicos():
    print("==== Tópicos na lista ====")
    for index, coleta in enumerate(coletas):
        print(f"{index+1}. {coleta.page}")
    print("==========================")

def mostrar_status():
    print("==== Status das coletas ====")
    for coleta in coletas:
        conn_thread = sqlite3.connect("wiki_data.db")
        cursor = conn_thread.cursor()
        cursor.execute("SELECT data FROM dados WHERE pagina=?", (coleta.page,))
        result = cursor.fetchone()
        if result:
            status = "Concluída"
        else:
            status = "Em andamento"
        print(f"Página: {coleta.page} - Status: {status}")
        conn_thread.close()
    print("==========================")

# Iniciar a coleta dos dados das páginas na lista
for page in paginas:
    coleta = WikipediaCollector(page)
    coleta.start()
    coletas.append(coleta)

# Configurar o modelo de perguntas e respostas
qa_model = pipeline("question-answering", model="neuralmind/bert-base-portuguese-cased",
                    tokenizer="neuralmind/bert-base-portuguese-cased", truncation="only_first")



class QADataset(Dataset):
    def __init__(self, questions, contexts, answers):
        self.questions = questions
        self.contexts = contexts
        self.answers = answers
        self.tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        encoded_input = self.tokenizer.encode_plus(
            self.questions[idx],
            self.contexts[idx],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        input_ids = encoded_input["input_ids"].squeeze()
        attention_mask = encoded_input["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "start_positions": self.answers[idx]["start_position"],
            "end_positions": self.answers[idx]["end_position"],
        }

def train_model():
    cursor = conn.cursor()
    cursor.execute("SELECT texto FROM dados")
    rows = cursor.fetchall()
    texts = [json.loads(row[0])["text"] for row in rows]

    questions = ["O que é inteligência artificial?", "Como funciona o aprendizado de máquina?",
                 "Qual é a importância do processamento de linguagem natural?"]
    answers = [
        {"start_position": 17, "end_position": 47},
        {"start_position": 25, "end_position": 54},
        {"start_position": 26, "end_position": 59},
    ]

    dataset = QADataset(questions, texts, answers)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = BertForQuestionAnswering.from_pretrained("neuralmind/bert-base-portuguese-cased")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-5)

    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions,
            )

            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch:", epoch + 1, "Loss:", total_loss)

    # Save the model
    model.save_pretrained("trained_model")

def buscar_dados():
    cursor = conn.cursor()
    cursor.execute("SELECT texto FROM dados")
    rows = cursor.fetchall()
    texts = [json.loads(row[0])["text"] for row in rows]
    return "\n".join(text for text in texts)

# Treinar o modelo de perguntas e respostas
train_model()

while True:
    print("==== MENU ====")
    print("1. Fazer uma pergunta")
    print("2. Adicionar um tópico para coleta")
    print("3. Mostrar status das coletas")
    print("4. Listar tópicos na lista")
    print("5. Verificar se passou por todos os tópicos")
    print("6. Atualizar banco de dados")
    print("7. Sair")

    escolha = input("Escolha uma opção: ")

    if escolha == "1":
        pergunta = input("Digite sua pergunta: ")
        context = buscar_dados()

        # Carregar o modelo treinado
        trained_model = BertForQuestionAnswering.from_pretrained("trained_model")
        trained_model.eval()

        tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

        encoded_input = tokenizer.encode_plus(
            pergunta,
            context,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]

        with torch.no_grad():
            outputs = trained_model(input_ids=input_ids, attention_mask=attention_mask)

        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        all_tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze())

        start_index = torch.argmax(start_scores)
        end_index = torch.argmax(end_scores)

        answer = " ".join(all_tokens[start_index:end_index + 1])
        answer = answer.replace(" ##", "")

        print("Resposta:", answer)

        # Salvar a pergunta e resposta no banco de dados
        cursor = conn.cursor()
        cursor.execute("INSERT INTO perguntas_respostas (pergunta, resposta) VALUES (?, ?)", (pergunta, answer))
        conn.commit()

    elif escolha == "2":
        listar_topicos()
        topico = input("Digite o nome do tópico para coleta: ")

        if topico in paginas:
            print("Tópico já está na lista de coleta.")
        else:
            paginas.append(topico)
            coleta = WikipediaCollector(topico)
            coleta.start()
            coletas.append(coleta)
            print("Tópico adicionado com sucesso.")
            coleta.join()

    elif escolha == "3":
        mostrar_status()


    elif escolha == "4":
        listar_topicos()

    elif escolha == "5":
        if len(coletas) == 0:
            print("Todos os tópicos foram coletados.")
        else:
            print("Ainda existem tópicos não coletados.")

    elif escolha == "6":
        atualizar_banco()
        print("Banco de dados atualizado.")

    elif escolha == "7":
        break

    else:
        print("Opção inválida. Por favor, escolha uma opção válida.")
