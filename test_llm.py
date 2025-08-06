from os import system
from models.model import LLMModel
import json
import lmstudio as lms
import argparse

parser = argparse.ArgumentParser(description="Run Test on a Language Model")

parser.add_argument("-m", "--model", type=str, default="g4b", help = "Short name of the model. Options: g4b (Gemma3:4B), g12 (Gemma3:12B), r1 (DeepSeek-R1)")

args: argparse.Namespace = parser.parse_args()



MODEL_PATH_LIST = "models/model_list.json"

with open(MODEL_PATH_LIST, "r") as file:
    model_list: list[dict] = json.load(file)

model_name: str = args.model

# Seleccionamos el diccionario del modelo
model_dict: dict[str, str] | None = next((model for model in model_list if model["short"] == model_name), None)

if model_dict is None:
    raise ValueError(f"Model '{model_name}' not found in model list.")
else: 
    model_key: str = model_dict["key"]
  

# Creamos una instancia del modelo
model: LLMModel = LLMModel(model_key)
print(model.llm)

system_prompt: str = """Responde en español. Eres un físico experto en física cuántica y relatividad. Responde a las preguntas de manera clara y concisa.
    Proporciona ejemplos cuando sea necesario. 
    Evita el uso de jerga técnica complicada."""
model.set_system_context(system_prompt)


print(model.chat)    
model.respond_stream("Hola! ¿Cuál es tu nombre?")
# model.respond_stream("¿Cuál es tu rol?")
# model.respond_stream("¿Qué es la física cuántica en 10 palabras?")