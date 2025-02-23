from .neuron import Neuron, ControllerNeuron

class NeuralNetwork:
  def __init__(self, input_size: int, hidden_layers: list[int], output_size: int):
    self.layers = []
    # Camada de entrada: cada neurônio recebe um valor de entrada diretamente
    input_layer = [Neuron(id=f"input_{i}") for i in range(input_size)]
    self.layers.append(input_layer)

    # Camadas ocultas
    layer_index = 1
    for num in hidden_layers:
      hidden_layer = []
      for i in range(num):
        # Aproximadamente 20% dos neurônios serão controladores (exemplo: i % 5 == 0)
        neuron = ControllerNeuron(id=f"layer{layer_index}_neuron{i}") if (i % 5 == 0) else Neuron(id=f"layer{layer_index}_neuron{i}")
        hidden_layer.append(neuron)
      self.layers.append(hidden_layer)
      layer_index += 1

    # Camada de saída
    output_layer = [Neuron(id=f"output_{i}") for i in range(output_size)]
    self.layers.append(output_layer)

    self.connect_layers()

  def connect_layers(self) -> None:
    # Conecta todas as camadas de forma totalmente conectada
    for l in range(len(self.layers) - 1):
      for neuron in self.layers[l]:
        for next_neuron in self.layers[l + 1]:
          neuron.add_connection(next_neuron)

  def feedforward(self, input_data: list[float]) -> list[float]:
    if len(input_data) != len(self.layers[0]):
      raise ValueError("O tamanho de input_data deve ser igual ao número de neurônios na camada de entrada.")
    # Propaga os sinais pela rede
    for i, neuron in enumerate(self.layers[0]):
      neuron.output = input_data[i]

    for l in range(1, len(self.layers)):
      prev_layer = self.layers[l - 1]
      inputs = {neuron.id: neuron.output for neuron in prev_layer}
      for neuron in self.layers[l]:
        neuron.activate(inputs)

    return [neuron.output for neuron in self.layers[-1]]