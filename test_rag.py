import os 
from models.document_store import DocumentStore
import json
import argparse
from models.model import LLModel

parser = argparse.ArgumentParser(description="Run Test on a Language Model")

parser.add_argument("-i", "--input", type=str, required=False, help = "Folder with txt documents")
parser.add_argument("-m", "--model", type=str, default="g4b", help = "Short name of the model. Options: g4b (Gemma3:4B), g12 (Gemma3:12B), r1 (DeepSeek-R1)")

args = parser.parse_args()

input_folder = args.input

text_contents = []
text_info_template = {"text":None, "metadata": None}

for file in os.listdir(input_folder):
  if file.endswith(".txt"):
    with open(os.path.join(input_folder, file), "r", encoding="utf-8") as f:
      text = f.read()
      text_info = text_info_template.copy()
      text_info["text"] = text
      text_info["metadata"] = {"file_name": file,
                               "content": file.replace(".txt", "")}
      text_contents.append(text_info)

dc = DocumentStore()
if args.input is not None:
  for documents in text_contents:
      dc.add_text_document(documents["text"], documents["metadata"])


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
model: LLModel = LLModel(model_key)
    
print("Introduce la consulta:")
query = input()

# Búsqueda RAG
results = dc.search_similar_documents(query=query, n_results=5)
# Guardamos copia de los resultados
json_results = json.dumps(results, indent=2, ensure_ascii=False)

context_prompt = " ".join([result["document"] for result in results])

with open("search_results.json", "w", encoding="utf-8") as f:
    f.write(json_results)
print("Resultados guardados en 'search_results.json'")


system_prompt: str = """Responde en español. Tu proósito es recuperar información relevante de documentos y actas con la actividad de una organización.
Para ello, se está utilizando tecnología RAG, de manera que recibiras un contexto. Si la información solicitada no se encuentra disponible en dicho contexto,
se deberá proporcionar información resumida del contexto, pero aclarar que no se sabe la respuesta. 
"""
model.set_system_context(system_prompt)

rag_prompt: str = """ Información recuperada:
    {context}
    
    Pregunta: {question}
    """

rag_prompt = rag_prompt.format(context=context_prompt, question=query)

model.respond_stream(rag_prompt, raw=False)