import random

class BaseModel:
  def __init__(self):
    self.config = {}

  def generate_response(self, prompt):
    raise NotImplementedError("Subclasses must implement generate_response method")

class QwenModel(BaseModel):
  def __init__(self, temperature=0.7):
    super().__init__()
    self.temperature = temperature

  def generate_response(self, prompt):
    # Simula uma resposta gerada pelo modelo Qwen baseado na temperatura
    responses = [
      "Esta é uma resposta simulada pelo QwenModel.",
      "O QwenModel processou seu prompt e retornou esta resposta criativa.",
      "Resposta do QwenModel: sua mensagem foi interpretada de forma única.",
      "QwenModel diz: exemplo de resposta gerada dinamicamente."
    ]
    return random.choice(responses)

class LlamaModel(BaseModel):
  def __init__(self, max_tokens=100):
    super().__init__()
    self.max_tokens = max_tokens

  def generate_response(self, prompt):
    # Simula uma resposta gerada pelo LlamaModel considerando o limite de tokens
    responses = [
      "Esta é uma resposta simulada pelo LlamaModel.",
      "O LlamaModel respondeu ao seu input com esta mensagem.",
      "Resposta do LlamaModel: exemplo de saída demonstrativa.",
      "LlamaModel diz: esta é uma resposta elaborada de exemplo."
    ]
    return random.choice(responses)