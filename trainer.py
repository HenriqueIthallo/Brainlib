import random
from .nlp_utils import tokenize, vectorize

class ChatbotTrainer:
  def __init__(self, neural_network):
    self.network = neural_network

  def build_vocab(self, conversation_pairs: list[tuple[str, str]]) -> tuple[list[str], dict]:
    all_tokens = []
    for inp, out in conversation_pairs:
      all_tokens.extend(tokenize(inp))
      all_tokens.extend(tokenize(out))
    vocab = list(set(all_tokens))
    word_to_index = {word: i for i, word in enumerate(vocab)}
    return vocab, word_to_index

  def train(self, conversation_pairs: list[tuple[str, str]], epochs: int = 10, learning_rate: float = 0.01) -> None:
    """
    Treina a rede neural utilizando pares de conversação.
    conversation_pairs: lista de tuplas (entrada, resposta)
    """
    vocab, word_to_index = self.build_vocab(conversation_pairs)
    vocab_size = len(vocab)
    if len(self.network.layers[0]) != vocab_size or len(self.network.layers[-1]) != vocab_size:
      print("Aviso: Dimensões da rede incompatíveis com o tamanho do vocabulário.")
      print(f"Vocabulário: {vocab_size}, Entrada: {len(self.network.layers[0])}, Saída: {len(self.network.layers[-1])}")

    for epoch in range(epochs):
      total_loss = 0.0
      random.shuffle(conversation_pairs)  # Embaralha os pares a cada época
      for inp, expected in conversation_pairs:
        input_vec = vectorize(inp, word_to_index, vocab_size)
        target_vec = vectorize(expected, word_to_index, vocab_size)
        output = self.network.feedforward(input_vec)
        loss = sum((o - t) ** 2 for o, t in zip(output, target_vec)) / len(target_vec)
        total_loss += loss
        # Atualização simplificada dos pesos na camada anterior à de saída
        prev_layer = self.network.layers[-2]
        for neuron in prev_layer:
          for key, conn in neuron.weights.items():
            gradient = loss * conn['weight']
            conn['weight'] -= learning_rate * gradient
      avg_loss = total_loss / len(conversation_pairs)
      print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

  def predict(self, input_text: str, word_to_index: dict, vocab_size: int) -> list[int]:
    input_vec = vectorize(input_text, word_to_index, vocab_size)
    output = self.network.feedforward(input_vec)
    # Seleciona os índices das palavras cuja saída excede um limiar (ex.: 0.5)
    predicted_indices = [i for i, val in enumerate(output) if val > 0.5]
    return predicted_indices

  def pretrain(self, db=None, epochs: int = 10, learning_rate: float = 0.01) -> None:
    from .pretraining_db import PretrainingDB
    if db is None:
      db = PretrainingDB()
    conversation_pairs = db.get_all_pairs()
    if not conversation_pairs:
      print("Nenhum par de conversação encontrado no banco de dados de pré-treinamento.")
      return
    print("Iniciando pré-treinamento com os dados do banco...")
    self.train(conversation_pairs, epochs, learning_rate)