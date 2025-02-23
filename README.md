# BrainLib - Biblioteca para Chatbots Inspirada no Cérebro (Versão Otimizada)

BrainLib é uma biblioteca Python inovadora que simula uma rede neural inspirada no funcionamento do cérebro humano, ideal para o treinamento de chatbots. A arquitetura utiliza neurônios individuais e neurônios controladores para modular a atividade, criando uma estrutura hierárquica e dinâmica.

## Recursos

- Simulação de rede neural com múltiplas camadas.
- Neurônios individuais e neurônios controladores para modulação dinâmica.
- Métodos para treinamento utilizando dados de conversação.
- Processamento de linguagem natural com NLTK.
- Suporte para modelos de geração de texto (ex.: QwenModel e LlamaModel).
- Banco de Dados de Pré-Treinamento utilizando SQLite.
- Otimizações recentes:
  - Modularização da construção do vocabulário.
  - Embaralhamento dos pares de treinamento em cada época.
  - Melhoria nas verificações de integridade (ex.: dimensões de entrada/saída).

## Dependências

- Python 3.x
- numpy
- nltk
- gymnasiun
- ctransformers (ou alternativas permitidas)

## Instalação

1. Clone o repositório:
   git clone https://github.com/brainlib/brainlib.git

2. Instale as dependências:
   pip install numpy nltk gymnasiun ctransformers

3. (Opcional) Configure os recursos do NLTK:
   python -c "import nltk; nltk.download('punkt')"

## Uso Básico

Exemplo de integração da BrainLib em um projeto de chatbot:
```python
from brainlib import NeuralNetwork, ChatbotTrainer

# Suponha que o tamanho do vocabulário seja 100 (ajuste conforme seus dados)
vocab_size = 100  
nn = NeuralNetwork(input_size=vocab_size, hidden_layers=[50, 30], output_size=vocab_size)
trainer = ChatbotTrainer(nn)

# Dados de conversação de exemplo
conversations = [
  ("Olá", "Oi, como posso ajudar?"),
  ("Qual seu nome?", "Eu sou BrainLib, seu assistente.")
]

trainer.train(conversations, epochs=10, learning_rate=0.01)
``` 