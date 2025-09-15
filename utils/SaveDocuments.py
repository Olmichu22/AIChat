import argparse
import json
import os
from models.document_store import DocumentStore

parser = argparse.ArgumentParser(description="Save documnents y RAG database.")
parser.add_argument("-i", "--input", type=str, required=True, help="Folder with txt documents")

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