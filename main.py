#!/usr/bin/env python3
# minichat_gradio_ui_mod.py

import gradio as gr
import requests
import base64
import io
import json

OLLAMA_URL = "http://localhost:11434/api/generate"

# Función para chat de texto
def chat_with_gemma(user_message, chat_history, model):
    if chat_history is None:
        chat_history = []
    # Montar prompt con historial
    prompt = "".join(f"Usuario: {u}\nAsistente: {a}\n" for u, a in chat_history)
    prompt += f"Usuario: {user_message}\nAsistente: "

    payload = {"model": model, "prompt": prompt}
    resp = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=60)
    resp.raise_for_status()
    # Reconstruir streaming
    reply_chunks = []
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            part = json.loads(line)
            reply_chunks.append(part.get("response", ""))
        except json.JSONDecodeError:
            continue

    reply = "".join(reply_chunks).strip()
    chat_history.append((user_message, reply))
    return "", chat_history, chat_history

# Función para caption de imagen
def caption_image(img, chat_history, model):
    if chat_history is None:
        chat_history = []
    # Convertir a base64
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    payload = {"model": model, "prompt": "Caption this image", "images": [b64]}
    resp = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=60)
    resp.raise_for_status()
    caption_chunks = []
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            part = json.loads(line)
            caption_chunks.append(part.get("response", ""))
        except json.JSONDecodeError:
            continue

    caption = "".join(caption_chunks).strip()
    chat_history.append(("[imagen]", caption))
    return chat_history, chat_history

# Función para limpiar chat
def clear_chat():
    return [], []

# CSS personalizado para ancho, alto y botones más pequeños
custom_css = """
#chatbot { width: 900px !important; height: 400px !important; }
#send_btn, #clear_btn { padding: 4px 8px !important; font-size: 14px !important; }
.gradio-textbox textarea { height: 150px !important; }
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("## MiniChat Web con Gemma (elige modelo)")

    # Selector de modelo
    model_dd = gr.Dropdown(
        choices=["gemma3:12b", "gemma3:latest"],
        value="gemma3:latest",
        label="Modelo"
    )

    # Chatbot y estado interno
    chatbot = gr.Chatbot(elem_id="chatbot")
    state = gr.State([])

    # Barra de entrada y botones
    with gr.Row():
        txt = gr.Textbox(
            placeholder="Escribe tu mensaje…",
            show_label=False,
            lines=3
        )
        send_btn = gr.Button("Enviar", elem_id="send_btn")
        clear_btn = gr.Button("Borrar chat", elem_id="clear_btn")

    # Componente de imagen para subir y procesar
    img_input = gr.Image(
        type="pil",
        label="📷 Sube una imagen para caption",
        elem_id="img_input"
    )

    # Conexiones de eventos
    send_btn.click(
        fn=chat_with_gemma,
        inputs=[txt, state, model_dd],
        outputs=[txt, state, chatbot]
    )
    clear_btn.click(
        fn=clear_chat,
        inputs=None,
        outputs=[state, chatbot]
    )
    img_input.upload(
        fn=caption_image,
        inputs=[img_input, state, model_dd],
        outputs=[state, chatbot]
    )

    demo.launch()
