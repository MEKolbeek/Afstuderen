# prototype_afstuderen.py
import os
import csv
import json
import re
import unicodedata
import warnings
from collections import OrderedDict

import fitz  # PyMuPDF
import requests

warnings.filterwarnings("ignore", category=UserWarning)

API_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3"

# ======================
# Prompts (statish, geen .format)
# ======================

BASE_PROMPT = (
    "Je bent rechercheur in een strafrechtelijk onderzoek.\n"
    "Je analyseert teksten uit een politiedossier, zoals processen-verbaal, verklaringen, tapgesprekken en observaties.\n\n"
    "Taken:\n"
    "- Zoek fragmenten die passen bij de zoekopdracht van de gebruiker.\n"
    "- Geef alleen letterlijke citaten uit de tekst. Geen interpretatie. Geen samenvatting.\n"
    "- Leg verbanden door losse citaten te selecteren, nooit door eigen tekst toe te voegen.\n"
    "- Behoud de originele spelling, grammatica en interpunctie.\n\n"
    "Regels:\n"
    "- Gebruik uitsluitend informatie die in de aangeleverde tekst staat.\n"
    "- Laat begeleidende zinnen of uitleg weg.\n"
    "- Als er geen relevante citaten zijn, geef niets terug.\n\n"
    "Output:\n"
    "- Gebruik JSON Lines. Eén object per regel. Geen extra tekst.\n"
    '- Per regel exact dit schema: {"citaat": "...", "pagina": <nummer of null>, "titel": "<documenttitel of pv-nummer>"}\n\n'
    "Context:\n"
    "- De tekst bevat paginamarkers in de vorm [Pagina N]. Gebruik deze om het veld 'pagina' te vullen waar mogelijk.\n"
    "- Dit is een juridisch dossier. Werk strikt en herleidbaar.\n"
)

ANSWER_HEADER = (
    "Je krijgt hieronder een gebruikersvraag en een set letterlijke citaten uit documenten. "
    "Schrijf een uitgebreid, samenhangend antwoord voor een officier van justitie. "
    "Gebruik alleen wat in de citaten staat. Geen speculatie. Geen informatie buiten de citaten. "
    "Verwijs in de lopende tekst kort naar documenttitel en paginanummer tussen haakjes, zoals (Titel, p. 12). "
    "Geen bulletlist tenzij noodzakelijk. Gewoon lopende tekst.\n\n"
)

SELECT_HEADER = (
    "Je krijgt de gebruikersvraag en het conceptantwoord plus alle gevonden citaten. "
    "Selecteer uitsluitend de citaten die daadwerkelijk iets toevoegen als bewijs voor het antwoord. "
    "Laat ruis weg. Kies beknopt maar volledig. Houd waar mogelijk variatie in bronnen.\n\n"
    "Geef als JSON Lines, exact per regel:\n"
    "{\"citaat\":\"...\", \"pagina\": null, \"titel\":\"...\"}\n"
    "Geen extra tekst.\n"
)

# ======================
# Low-level helpers
# ======================

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # Houd het simpel, verwijder exotische combinaties om CSV te sparen
    s = unicodedata.normalize("NFKC", s)
    return s.strip()

def post_chat(messages, timeout=180, stream=False) -> str:
    """
    Robuuste chat-call naar Ollama /api/chat zonder stream hangers.
    Verwacht non-stream JSON: {"message": {"content": "..."}}
    """
    payload = {"model": MODEL_NAME, "messages": messages, "stream": stream}
    r = requests.post(API_URL, json=payload, timeout=timeout, stream=stream)
    r.raise_for_status()
    if not stream:
        js = r.json()
        msg = js.get("message", {}).get("content", "")
        return normalize_text(msg)
    # fallback stream (niet default gebruikt)
    out = ""
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            js = json.loads(line)
        except json.JSONDecodeError:
            continue
        if js.get("done"):
            break
        part = js.get("message", {}).get("content", "")
        if part:
            out += part
    return normalize_text(out)

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += f"[Pagina {page.number + 1}]\n"
            text += page.get_text()
    return text

def get_next_export_filename(directory, base_name="bewijscitaten", extension=".csv") -> str:
    counter = 1
    while True:
        fn = f"{base_name}_{counter}{extension}"
        fp = os.path.join(directory, fn)
        if not os.path.exists(fp):
            return fp
        counter += 1

# ======================
# Parsing en schoonmaak
# ======================

def parse_jsonl(raw: str, pdf_name: str) -> list:
    out = []
    for line in raw.splitlines():
        s = line.strip().rstrip(",")
        if not s or not s.startswith("{") or not s.endswith("}"):
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            continue
        citaat = str(obj.get("citaat", "")).strip()
        pagina = obj.get("pagina", None)
        titel = str(obj.get("titel", "")).strip() or pdf_name
        if citaat:
            # pagina naar int of None
            if isinstance(pagina, str):
                pagina = pagina.strip()
                if pagina == "" or pagina.lower() == "null":
                    pagina = None
                elif pagina.isdigit():
                    pagina = int(pagina)
                else:
                    pagina = None
            elif isinstance(pagina, (int, float)):
                pagina = int(pagina)
            else:
                pagina = None
            out.append({"citaat": citaat, "pagina": pagina, "titel": titel, "document": pdf_name})
    return out

def regex_fallback(raw: str, pdf_name: str) -> list:
    out = []
    for line in raw.splitlines():
        t = line.strip()
        if not t:
            continue
        low = t.lower()
        if low.startswith(("hier zijn", "onderstaande", "lijst met", "opdracht", "citaten:")):
            continue
        pagina = None
        m = re.search(r"\[Pagina\s*(\d+)\]", t)
        if m:
            pagina = int(m.group(1))
            t = t.replace(m.group(0), "").strip()
        out.append({"citaat": t, "pagina": pagina, "titel": pdf_name, "document": pdf_name})
    return out

def parse_output(raw: str, pdf_name: str) -> list:
    js = parse_jsonl(raw, pdf_name)
    if js:
        return js
    return regex_fallback(raw, pdf_name)

def dedup_citaten(items: list) -> list:
    seen = OrderedDict()
    for it in items:
        key = re.sub(r"\s+", " ", it["citaat"]).strip()
        if key not in seen:
            seen[key] = it
    return list(seen.values())

def filter_kleine_ruis(items: list, min_len: int = 25) -> list:
    out = []
    for it in items:
        c = it["citaat"].strip()
        if len(c) < min_len:
            continue
        if re.match(r"^hier\s+zijn|^onderstaande|^opdracht", c.lower()):
            continue
        out.append(it)
    return out

def chunk_text(items: list, max_chars: int = 14000) -> str:
    # Eén citaat per regel, met bronvermelding
    lines = []
    total = 0
    for it in items:
        pagina_txt = str(it["pagina"]) if it["pagina"] is not None else "?"
        line = f'- "{it["citaat"]}"  [{it["titel"]}, p. {pagina_txt}]'
        add = len(line) + 1
        if total + add > max_chars:
            break
        lines.append(line)
        total += add
    return "\n".join(lines)

# ======================
# Promptbouw (veilig zonder .format)
# ======================

def build_doc_prompt(pdf_name: str, user_query: str, pdf_text: str) -> str:
    return (
        f"Document: {pdf_name}\n\n"
        f"{BASE_PROMPT}\n"
        f'Zoekopdracht: "{user_query}"\n\n'
        f"TEKST START\n{pdf_text}\nTEKST EINDE"
    )

def build_answer_prompt(vraag: str, citaten_blok: str) -> str:
    return (
        ANSWER_HEADER
        + "VRAAG:\n" + vraag + "\n\n"
        + "CITATEN:\n" + citaten_blok + "\n"
    )

def build_select_prompt(vraag: str, answer: str, citaten_blok: str) -> str:
    return (
        SELECT_HEADER
        + "VRAAG:\n" + vraag + "\n\n"
        + "ANTWOORD:\n" + answer + "\n\n"
        + "CITATEN:\n" + citaten_blok + "\n"
    )

# ======================
# Pipeline-stappen
# ======================

def extract_citaten_from_dir(pdf_dir: str, vraag: str) -> list:
    pdfs = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    alle = []
    for pdf in pdfs:
        path = os.path.join(pdf_dir, pdf)
        print(f"Verwerken: {pdf}")
        try:
            text = extract_text_from_pdf(path)
        except Exception as e:
            print(f"[!] Kon {pdf} niet lezen: {e}")
            continue
        prompt = build_doc_prompt(os.path.basename(pdf), vraag, text)
        try:
            raw = post_chat([{"role": "user", "content": prompt}], timeout=240, stream=False)
        except requests.RequestException as e:
            print(f"[!] API-fout bij {pdf}: {e}")
            continue
        parsed = parse_output(raw, pdf)
        alle.extend(parsed)
    alle = filter_kleine_ruis(dedup_citaten(alle))
    return alle

def synthesize_answer(vraag: str, citaten: list) -> str:
    if not citaten:
        return "Er zijn geen relevante citaten gevonden voor deze vraag."
    blok = chunk_text(citaten, max_chars=14000)
    prompt = build_answer_prompt(vraag, blok)
    ans = post_chat([{"role": "user", "content": prompt}], timeout=240, stream=False)
    return ans.strip()

def select_supporting_quotes(vraag: str, answer: str, citaten: list) -> list:
    if not citaten:
        return []
    blok = chunk_text(citaten, max_chars=14000)
    prompt = build_select_prompt(vraag, answer, blok)
    raw = post_chat([{"role": "user", "content": prompt}], timeout=240, stream=False)
    gekozen = parse_jsonl(raw, pdf_name="")
    if not gekozen:
        # Back-up: neem een beknopte set
        gekozen = citaten[:10]
        for g in gekozen:
            g.pop("document", None)
    return gekozen

def write_csv(csv_path: str, citaten: list, titel2doc: dict):
    # Kolomvolgorde: citaat, titel, document, pagina
    veldnamen = ["citaat", "titel", "document", "pagina"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=veldnamen, delimiter=",")
        w.writeheader()
        for it in citaten:
            titel = it.get("titel", "")
            document = it.get("document") or titel2doc.get(titel, titel)
            pagina = it.get("pagina")
            row = {
                "citaat": it.get("citaat", ""),
                "titel": titel,
                "document": document,
                "pagina": pagina if pagina is not None else ""
            }
            w.writerow(row)

# ======================
# Main
# ======================

def main(pdf_dir: str, vraag: str):
    # 1) Alle citaten
    alle_citaten = extract_citaten_from_dir(pdf_dir, vraag)

    # mapping voor documentherstel
    titel2doc = {}
    for it in alle_citaten:
        titel2doc.setdefault(it["titel"], it["document"])

    # 2) Antwoord voor terminal
    antwoord = synthesize_answer(vraag, alle_citaten)

    # 3) Selectie bewijscitaten
    selectie = select_supporting_quotes(vraag, antwoord, alle_citaten)
    # herstel documentnaam waar nodig
    for it in selectie:
        if not it.get("document"):
            it["document"] = titel2doc.get(it.get("titel", ""), it.get("titel", ""))

    # 4) CSV schrijven
    csv_path = get_next_export_filename(pdf_dir, base_name="bewijscitaten", extension=".csv")
    write_csv(csv_path, selectie, titel2doc)

    # 5) Terminal output
    print("\n=== ANTWOORD ===\n")
    print(antwoord)
    print("\n=== CSV met bewijscitaten ===")
    print(csv_path)

if __name__ == "__main__":
    # Pas deze twee aan
    PDF_DIR = "/Users/116299/Documents/Project_Afstuderen/test"
    USER_QUERY = "Welke plaatsnamen worden genoemd en corresponderen zij met strafbare feiten?"
    try:
        main(PDF_DIR, USER_QUERY)
    except KeyboardInterrupt:
        print("\n[!] Afgebroken door gebruiker.")
