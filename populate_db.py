#!/usr/bin/env python3
from brainlib.pretraining_db import PretrainingDB

def populate_db(num_entries=1000):
  db = PretrainingDB()
  current_count = db.count_pairs()
  if current_count >= num_entries:
    print(f"O banco de dados já possui {current_count} pares de conversação. Nenhuma ação necessária.")
    db.close()
    return

  missing = num_entries - current_count
  sample_pairs = [
    ("Qual é a capital da França?", "Paris"),
    ("Qual é o maior planeta do sistema solar?", "Júpiter"),
    ("Quem foi o primeiro presidente dos Estados Unidos?", "George Washington"),
    ("Quanto é 2+2?", "4"),
    ("Qual é a cor do céu em um dia ensolarado?", "Azul")
  ]

  for i in range(missing):
    question, answer = sample_pairs[i % len(sample_pairs)]
    # Garante a singularidade adicionando um índice
    question_unique = f"{question} (Exemplo {current_count + i + 1})"
    answer_unique = f"{answer} (Exemplo {current_count + i + 1})"
    db.add_conversation_pair(question_unique, answer_unique)

  total = db.count_pairs()
  print(f"Banco de dados populado com {missing} pares de conversação. Total agora: {total} pares.")
  db.close()

if __name__ == '__main__':
  populate_db()