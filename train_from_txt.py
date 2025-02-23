#!/usr/bin/env python3
import os
from brainlib.neural_network import NeuralNetwork
from brainlib.trainer import ChatbotTrainer
from brainlib.nlp_utils import tokenize

def load_training_data(file_path: str):
  pairs = []
  if not os.path.exists(file_path):
    raise FileNotFoundError(f"Arquivo {file_path} não encontrado.")
  with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read().strip()
  lines = content.splitlines()
  i = 0
  while i < len(lines):
    line = lines[i].strip()
    if line.startswith("Pergunta"):
      if i + 1 < len(lines):
        resp_line = lines[i + 1].strip()
        # Remove os prefixos "Pergunta <num>:" e "Resposta <num>:"
        question = line.split(":", 1)[-1].strip()
        response = resp_line.split(":", 1)[-1].strip()
        pairs.append((question, response))
        i += 2
      else:
        i += 1
    else:
      i += 1
  return pairs

def main():
  data_file = "training_data.txt"
  pairs = load_training_data(data_file)
  print(f"Carregado {len(pairs)} pares de treinamento.")
  
  # Constrói o vocabulário a partir dos dados carregados
  vocab = set()
  for question, response in pairs:
    vocab.update(tokenize(question))
    vocab.update(tokenize(response))
  vocab_size = len(vocab)
  print(f"Tamanho do vocabulário: {vocab_size}")
  
  # Cria a rede neural com dimensões baseadas no vocabulário
  nn = NeuralNetwork(input_size=vocab_size, hidden_layers=[50, 30], output_size=vocab_size)
  trainer = ChatbotTrainer(nn)
  
  # Treina o modelo com os pares carregados (exemplo usando 5 épocas)
  trainer.train(pairs, epochs=5, learning_rate=0.01)

if __name__ == '__main__':
  main()