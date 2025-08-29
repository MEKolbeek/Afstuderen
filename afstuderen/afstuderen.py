import os
import csv
import fitz  # PyMuPDF
import requests
import json
import unicodedata
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import re 

API_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3"

#paginanummering gaat nog steeds niet goed, klopt dit kan hier nog iets in veranderen
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += f"[Pagina {page.number + 1}]\n"
            text += page.get_text()
    return text

def normalize_text(text):
    if isinstance(text, str):
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        return text.strip()
    return ""

def query_llama(prompt):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=60, stream=True)
        response.raise_for_status()
        lines = response.iter_lines(decode_unicode=True)
        full_text = ""
        for line in lines:
            if line.strip():
                try:
                    json_line = json.loads(line)
                    if "message" in json_line and "content" in json_line["message"]:
                        full_text += json_line["message"]["content"]
                except json.JSONDecodeError:
                    print(f"[!] JSON-fout bij regel: {line}")
        return normalize_text(full_text.strip())
    except requests.exceptions.RequestException as e:
        return f"Fout: {e}"
    except json.JSONDecodeError:
        return "JSON decode error: Ongeldige serverrespons"

def build_prompt(variant, pdf_name, pdf_text):
    if variant == 1:
        return (f"Document: {pdf_name}\n\n"
                f"Tekst:\n{pdf_text}\n\n"
                "Maak een lijst met exacte citaten uit het document over crimineel gebruik en misbruik van vervoer en logistiek. Vermeld bij elk citaat de pagina.")
    elif variant == 2:
        return (f"Document: {pdf_name}\n\n"
                f"Tekst:\n{pdf_text}\n\n"
                "Je helpt bij het maken van een veiligheidsbeeld over crimineel gebruik en misbruik van vervoer en logistiek. "
                "Maak een lijst met exacte citaten die hierover gaan. Zet bij elk citaat de titel van het document en de pagina.")
    elif variant == 3:
        return (f"Document: {pdf_name}\n\n"
                f"Tekst:\n{pdf_text}\n\n"
                "Zet bovenaan de titel van het document. Maak een lijst van citaten die letterlijk uit het rapport zijn overgenomen en direct"
                "betrekking hebben op crimineel gebruik en misbruik van vervoer en logistiek. Voeg bij elk citaat de exacte pagina toe. Geen interpretatie."
                "Geen samenvatting. Alleen letterlijk gekopieerde tekst.")
    else:
        return ""

def parse_output(raw_output, variant):
    parsed = []

    lines = raw_output.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue

        pagina = ""
        titel = ""
        citaat = line

        # Paginanummer zoeken met regex (gaat nog niet goed, misschien andere regex search van maken? iets van in bottom zoeken oid)
        match = re.search(r'[Pp]agina[:\s]*([0-9]{1,3})', line)
        if match:
            pagina = match.group(1)
            citaat = line.replace(match.group(0), '').strip()

        # Titel zoeken bij variant 2 (optioneel, uit regel of begin)
        if variant == 2:
            titel_match = re.search(r'^(.+?)\s*[-–]\s*[Pp]agina', line)
            if titel_match:
                titel = titel_match.group(1).strip()
            elif "document" in line.lower():
                titel = "titel vermeld in citaat"
        
        parsed.append({
            "citaat": citaat,
            "pagina": pagina,
            "titel": titel if variant == 2 else ""
        })

    return parsed

def process_pdf(pdf_path, variant):
    text = extract_text_from_pdf(pdf_path)
    prompt = build_prompt(variant, os.path.basename(pdf_path), text)
    result = query_llama(prompt)
    return result

def get_next_export_filename(directory, base_name="export", extension=".
