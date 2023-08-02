from transformers import BertTokenizer, BertForQuestionAnswering
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW
import torch

# Classe do conjunto de dados personalizado para treinamento do modelo
class QADataset(Dataset):
    def __init__(self, questions, contexts, answers):
        self.questions = questions
        self.contexts = contexts
        self.answers = answers
        self.tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        question = self.questions[index]
        context = self.contexts[index]
        answer = self.answers[index]

        inputs = self.tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "start_positions": torch.tensor(answer["start"]),
            "end_positions": torch.tensor(answer["end"]),
        }

# Dados de treinamento de exemplo
exemplos_perguntas_respostas = [
    {"pergunta": "O que é inteligência artificial?", "resposta": "Inteligência artificial (IA) é o campo da ciência da computação que se concentra na criação de programas e sistemas que podem executar tarefas que normalmente exigiriam inteligência humana."},
    {"pergunta": "Quais são os principais subcampos da IA?", "resposta": "Alguns dos principais subcampos da IA incluem aprendizado de máquina, visão computacional, processamento de linguagem natural e robótica."},
    {"pergunta": "Como funciona o aprendizado de máquina?", "resposta": "O aprendizado de máquina é uma abordagem da IA que permite que os sistemas aprendam a partir de dados, identifiquem padrões e façam previsões ou tomem decisões sem serem explicitamente programados."},
    {"pergunta": "O que é processamento de linguagem natural?", "resposta": "Processamento de linguagem natural (PLN) é o campo da IA que se concentra na interação entre computadores e seres humanos por meio da linguagem humana. Ele permite que os computadores compreendam, interpretem e gerem texto ou fala."},
]

# Preparar os dados de treinamento
questions = [exemplo["pergunta"] for exemplo in exemplos_perguntas_respostas]
contexts = [context for context in buscar_dados()]  # Supondo que buscar_dados() retorna os textos coletados
answers = [{"start": context.find(exemplo["resposta"]), "end": context.find(exemplo["resposta"]) + len(exemplo["resposta"])} for exemplo, context in zip(exemplos_perguntas_respostas, contexts)]

dataset = QADataset(questions, contexts, answers)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Carregar o modelo pré-treinado
model = BertForQuestionAnswering.from_pretrained("neuralmind/bert-base-portuguese-cased")

# Definir os parâmetros de treinamento
optimizer = AdamW(model.parameters(), lr=1e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Realizar o treinamento
for epoch in range(5):
    model.train()
    total_loss = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs.loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}: Average Loss: {total_loss / len(dataloader)}")

# Salvando o modelo treinado
model.save_pretrained("*/bard")

# Agora você pode carregar o modelo treinado posteriormente para fazer inferências