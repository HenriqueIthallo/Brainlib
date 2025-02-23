import numpy as np
from typing import Optional, Dict, Any

class Neuron:
  def __init__(self, id: str, is_controller: bool = False) -> None:
    self.id = id
    self.is_controller = is_controller
    self.weights: Dict[str, Dict[str, Any]] = {}  # Mapeia o ID do neurônio alvo para um dicionário com o objeto e o peso
    self.bias = np.random.randn() * 0.1
    self.output = 0.0

  def add_connection(self, target_neuron: "Neuron", weight: Optional[float] = None) -> None:
    if weight is None:
      weight = np.random.randn() * 0.1
    self.weights[target_neuron.id] = {'neuron': target_neuron, 'weight': weight}

  def sigmoid(self, x: float) -> float:
    return 1 / (1 + np.exp(-x))

  def activate(self, inputs: dict) -> float:
    total = 0.0
    for conn in self.weights.values():
      input_val = inputs.get(conn['neuron'].id, 0)
      total += input_val * conn['weight']
    total += self.bias
    self.output = self.sigmoid(total)
    return self.output

  def update_weight(self, target_neuron_id: str, gradient: float, learning_rate: float = 0.01) -> None:
    if target_neuron_id in self.weights:
      self.weights[target_neuron_id]['weight'] -= learning_rate * gradient

class ControllerNeuron(Neuron):
  def __init__(self, id: str) -> None:
    super().__init__(id, is_controller=True)

  def modulate(self, target_neuron: "Neuron", factor: float) -> None:
    # Modula o viés do neurônio alvo de forma simples
    target_neuron.bias += factor