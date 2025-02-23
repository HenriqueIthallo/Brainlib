import nltk
from nltk.tokenize import word_tokenize
from typing import List, Dict

# Verifica se os recursos de tokenização estão disponíveis; caso contrário, faz o download.
try:
  nltk.data.find('tokenizers/punkt')
except LookupError:
  nltk.download('punkt')

def tokenize(text: str) -> List[str]:
  """
  Tokeniza o texto utilizando NLTK.
  """
  return word_tokenize(text, language='portuguese')

def vectorize(text: str, word_to_index: Dict[str, int], vocab_size: int) -> List[int]:
  """
  Converte o texto em um vetor binário de tamanho vocab_size.
  Cada posição recebe 1 se a palavra correspondente estiver presente.
  """
  tokens = tokenize(text.lower())
  vector = [0] * vocab_size
  for token in tokens:
    if token in word_to_index:
      vector[word_to_index[token]] = 1
  return vector