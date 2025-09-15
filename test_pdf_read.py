#!/usr/bin/env python3
import sys
import os
import PyPDF2

def read_pdf_raw(pdf_path: str) -> list[str]:
    """
    Lee un PDF con PyPDF2 y devuelve una lista con el texto bruto de cada página.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"No se encontró el PDF: {pdf_path}")

    reader = PyPDF2.PdfReader(pdf_path)

    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")
        except Exception as e:
            print(f"⚠️  No se pudo desencriptar el PDF: {e}")

    pages_text = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text()
            pages_text.append("" if txt is None else txt)
        except Exception as e:
            pages_text.append(f"<<ERROR extrayendo página {i+1}: {e}>>")
    return pages_text


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python leer_pdf.py <ruta_al_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    try:
        pages = read_pdf_raw(pdf_path)
        for i, text in enumerate(pages, start=1):
            print("=" * 40)
            print(f"📄 Página {i}")
            print("=" * 40)
            print(text)
            print("\n")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
