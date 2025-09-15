# prototype_afstuderen.py
import os
import csv
import json
import re
import unicodedata
import warnings
from collections import OrderedDict
from difflib import SequenceMatcher

import fitz  # PyMuPDF
import requests

warnings.filterwarnings("ignore", category=UserWarning)

API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/chat")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3")

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
    s = unicodedata.normalize("NFKC", s)
    return s.strip()

def normalize_for_match(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00A0", " ")
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def post_chat(messages, timeout=180, stream=False) -> str:
    payload = {"model": MODEL_NAME, "messages": messages, "stream": stream}
    r = requests.post(API_URL, json=payload, timeout=timeout, stream=stream)
    r.raise_for_status()
    if not stream:
        js = r.json()
        msg = js.get("message", {}).get("content", "")
        return normalize_text(msg)
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

# ======================
# PDF extractie & structuur
# ======================

def _split_paragraphs(page_text: str) -> list:
    raw_paras = re.split(r"\n{2,}", page_text)
    paras = []
    for p in raw_paras:
        t = p.strip()
        if not t:
            continue
        if len(t) < 20:
            continue
        paras.append(t)
    if not paras:
        lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
        chunk, out = [], []
        for ln in lines:
            chunk.append(ln)
            if len(" ".join(chunk)) >= 200:
                out.append(" ".join(chunk))
                chunk = []
        if chunk:
            out.append(" ".join(chunk))
        paras = out
    return paras

def _guess_pv_number_from_text(text: str) -> str:
    candidates = []
    patterns = [
        r"\bPV[\s\-.:]*nr[\s\-.:]*([A-Z0-9\-/.]{5,})",
        r"\bPV[\s\-.:]*nummer[\s\-.:]*([A-Z0-9\-/.]{5,})",
        r"\bProces\s*\-?\s*verbaal\s*nummer[\s\-.:]*([A-Z0-9\-/.]{5,})",
        r"\bPL[\s\-]?[A-Z0-9]{2,}[-/][A-Z0-9\-/.]{3,}",
        r"\bPV[\s\-]?[0-9]{6,}\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            pv = m.group(1) if m.lastindex else m.group(0)
            candidates.append(pv)
    if candidates:
        candidates.sort(key=len, reverse=True)
        return candidates[0].strip().strip(":. ;,")
    return ""

def _guess_title_from_meta(doc: fitz.Document) -> str:
    try:
        meta = doc.metadata or {}
    except Exception:
        meta = {}
    title = (meta.get("title") or meta.get("Title") or "").strip()
    return title

def extract_document_structure(pdf_path: str) -> dict:
    pdf_name = os.path.basename(pdf_path)
    paragraphs = []
    combined_text_parts = []
    first_pages_text = []

    with fitz.open(pdf_path) as doc:
        guessed_title = _guess_title_from_meta(doc)
        for page in doc:
            page_num = page.number + 1
            page_text = page.get_text() or ""
            if page_num <= 3:
                first_pages_text.append(page_text)
            combined_text_parts.append(f"[Pagina {page_num}]\n")
            combined_text_parts.append(page_text)
            paras = _split_paragraphs(page_text)
            for idx, p in enumerate(paras, start=1):
                paragraphs.append({
                    "page": page_num,
                    "paragraph": idx,
                    "text": p,
                    "norm": normalize_for_match(p),
                })

    doc_text = "".join(combined_text_parts)
    head_text = "\n".join(first_pages_text)

    pv_from_text = _guess_pv_number_from_text(head_text)
    pv_from_name = _guess_pv_number_from_text(pdf_name)
    pv_nummer = pv_from_text or pv_from_name

    titel = guessed_title or pv_nummer or os.path.splitext(pdf_name)[0]

    return {
        "pdf_name": pdf_name,
        "doc_text": doc_text,
        "paragraphs": paragraphs,
        "pv_nummer": pv_nummer,
        "titel": titel,
    }

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

def _best_fuzzy_match(citaat_norm: str, paras: list):
    best = None
    best_score = 0.0
    for p in paras:
        score = SequenceMatcher(a=citaat_norm, b=p["norm"]).ratio()
        if score > best_score:
            best_score = score
            best = p
    if best and best_score >= 0.92:
        return best
    return None

def locate_citaat(citaat: str, paragraphs: list):
    if not citaat:
        return None, None
    c_norm = normalize_for_match(citaat)
    for p in paragraphs:
        if c_norm and c_norm in p["norm"]:
            return p["page"], p["paragraph"]
    best = _best_fuzzy_match(c_norm, paragraphs)
    if best:
        return best["page"], best["paragraph"]
    return None, None

def chunk_text(items: list, max_chars: int = 14000) -> str:
    lines = []
    total = 0
    for it in items:
        pagina_txt = str(it.get("pagina")) if it.get("pagina") is not None else "?"
        titel_txt = it.get("titel") or it.get("document") or ""
        para = it.get("paragraaf")
        para_txt = f", ¶ {para}" if para is not None else ""
        line = f'- "{it["citaat"]}"  [{titel_txt}, p. {pagina_txt}{para_txt}]'
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

def extract_citaten_from_dir(pdf_dir: str, vraag: str):
    pdfs = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    alle = []
    docinfo_map = {}

    for pdf in pdfs:
        path = os.path.join(pdf_dir, pdf)
        print(f"Verwerken: {pdf}")
        try:
            docinfo = extract_document_structure(path)
            docinfo_map[pdf] = docinfo
        except Exception as e:
            print(f"[!] Kon {pdf} niet lezen: {e}")
            continue

        prompt = build_doc_prompt(docinfo["titel"], vraag, docinfo["doc_text"])
        try:
            raw = post_chat([{"role": "user", "content": prompt}], timeout=240, stream=False)
        except requests.RequestException as e:
            print(f"[!] API-fout bij {pdf}: {e}")
            continue

        parsed = parse_output(raw, pdf)
        for it in parsed:
            if not it.get("titel") or it["titel"] == pdf:
                it["titel"] = docinfo["titel"]
            it["document"] = pdf
        alle.extend(parsed)

    alle = filter_kleine_ruis(dedup_citaten(alle))

    # Backfill pagina/paragraaf + pv_nummer
    for it in alle:
        pdf_name = it.get("document")
        info = docinfo_map.get(pdf_name) or {}
        if it.get("pagina") is None and info.get("paragraphs"):
            page, para = locate_citaat(it["citaat"], info["paragraphs"])
            if page is not None:
                it["pagina"] = page
            if para is not None:
                it["paragraaf"] = para
        else:
            if info.get("paragraphs"):
                page, para = locate_citaat(it["citaat"], info["paragraphs"])
                if para is not None:
                    it["paragraaf"] = para
        if info.get("pv_nummer"):
            it["pv_nummer"] = info["pv_nummer"]
        if not it.get("titel") and info.get("titel"):
            it["titel"] = info["titel"]

    return alle, docinfo_map

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
        gekozen = citaten[:10]
        for g in gekozen:
            g.pop("document", None)
    index_by_quote = {normalize_for_match(c["citaat"]): c for c in citaten}
    enriched = []
    for g in gekozen:
        key = normalize_for_match(g.get("citaat", ""))
        base = index_by_quote.get(key, {})
        merged = {
            "citaat": g.get("citaat") or base.get("citaat"),
            "pagina": g.get("pagina", base.get("pagina")),
            "titel": g.get("titel") or base.get("titel"),
            "document": base.get("document"),
            "paragraaf": base.get("paragraaf"),
            "pv_nummer": base.get("pv_nummer"),
        }
        enriched.append(merged)
    return enriched

def write_csv(csv_path: str, citaten: list, titel2doc: dict):
    veldnamen = ["citaat", "titel", "pv_nummer", "document", "pagina", "paragraaf"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=veldnamen, delimiter=",")
        w.writeheader()
        for it in citaten:
            titel = it.get("titel", "")
            document = it.get("document") or titel2doc.get(titel, titel)
            pagina = it.get("pagina")
            paragraaf = it.get("paragraaf")
            row = {
                "citaat": it.get("citaat", ""),
                "titel": titel,
                "pv_nummer": it.get("pv_nummer", ""),
                "document": document,
                "pagina": pagina if pagina is not None else "",
                "paragraaf": paragraaf if paragraaf is not None else "",
            }
            w.writerow(row)

# ======================
# Main
# ======================

def main(pdf_dir: str, vraag: str):
    alle_citaten, docinfo_map = extract_citaten_from_dir(pdf_dir, vraag)

    titel2doc = {}
    for pdf_name, info in docinfo_map.items():
        titel2doc.setdefault(info.get("titel", pdf_name), pdf_name)

    antwoord = synthesize_answer(vraag, alle_citaten)

    selectie = select_supporting_quotes(vraag, antwoord, alle_citaten)
    for it in selectie:
        if not it.get("document"):
            it["document"] = titel2doc.get(it.get("titel", ""), it.get("titel", ""))

    csv_path = get_next_export_filename(pdf_dir, base_name="bewijscitaten", extension=".csv")
    write_csv(csv_path, selectie, titel2doc)

    print("\n=== ANTWOORD ===\n")
    print(antwoord)
    print("\n=== CSV met bewijscitaten ===")
    print(csv_path)

if __name__ == "__main__":
    PDF_DIR = os.getenv("PDF_DIR", "/Users/116299/Documents/Project_Afstuderen/test")
    USER_QUERY = os.getenv("USER_QUERY", "Welke plaatsnamen worden genoemd en corresponderen zij met strafbare feiten?")
    try:
        main(PDF_DIR, USER_QUERY)
    except KeyboardInterrupt:
        print("\n[!] Afgebroken door gebruiker.")
