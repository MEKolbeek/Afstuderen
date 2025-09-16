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
    '{"citaat":"...", "pagina": null, "titel":"..."}\n'
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
# Context packs helpers
# ======================

def get_corpus_sample_text(pdf_dir: str, max_pages: int = 2, max_chars: int = 30000) -> str:
    sample = []
    try:
        pdfs = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
        for pdf in pdfs:
            path = os.path.join(pdf_dir, pdf)
            try:
                with fitz.open(path) as doc:
                    for i, page in enumerate(doc):
                        if i >= max_pages:
                            break
                        sample.append(page.get_text() or "")
                        if sum(len(s) for s in sample) >= max_chars:
                            return "\n".join(sample)[:max_chars]
            except Exception:
                continue
    except Exception:
        pass
    return "\n".join(sample)[:max_chars]

def load_context_packs(ctx_dir: str, user_query: str, corpus_sample_text: str):
    index_path = os.path.join(ctx_dir, "index.json")
    common_path = os.path.join(ctx_dir, "common.json")
    if not (os.path.exists(index_path) and os.path.exists(common_path)):
        return None, []
    with open(index_path, "r", encoding="utf-8") as f:
        idx = json.load(f)
    with open(common_path, "r", encoding="utf-8") as f:
        common = json.load(f)

    active = []
    uq = (user_query or "").lower()
    for p in idx.get("packs", []):
        pack_path = os.path.join(ctx_dir, p.get("path", ""))
        if not os.path.exists(pack_path):
            continue
        try:
            with open(pack_path, "r", encoding="utf-8") as f:
                pack = json.load(f)
        except Exception:
            continue
        triggers = [t.lower() for t in pack.get("mapping_triggers", [])]
        query_hit = any(t in uq for t in triggers)
        corpus_hit = any(re.search(rf"\\b{re.escape(t)}\\b", corpus_sample_text, flags=re.I) for t in triggers[:20])
        if query_hit or corpus_hit:
            active.append(pack)
    return common, active

def build_context_hints(active_packs: list, max_hints: int = 100) -> list:
    hints = []
    for pack in active_packs:
        for item in pack.get("ontology", []):
            hints.extend(item.get("synonyms", [])[:25])
            hints.extend(item.get("indicators", [])[:10])
        if "token_terms" in pack:
            hints.extend(pack["token_terms"][:20])
        if "cex_dex_terms" in pack:
            hints.extend(pack["cex_dex_terms"][:20])
    seen = set()
    out = []
    for h in hints:
        h = str(h).strip()
        if not h or h.lower() in seen:
            continue
        seen.add(h.lower())
        out.append(h)
        if len(out) >= max_hints:
            break
    return out

# ======================
# Volatile context extractie (LLM + regex)
# ======================

LICENSE_PLATE_PATTERNS = [
    r"\b[A-Z]{2}-\d{3}-[A-Z]\b",
    r"\b\d{2}-[A-Z]{3}-\d\b",
    r"\b[A-Z]{2}-\d{2}-[A-Z]{2}\b",
    r"\b\d{2}-[A-Z]{2}-\d{2}\b",
    r"\b[A-Z]{2}-\d{2}-\d{2}\b",
    r"\b\d{3}-[A-Z]{2}-[A-Z]\b",
]

PHONE_PATTERNS = [
    r"\b(?:\+31\s?6\s?\d{8}|06[-\s]?\d{8})\b",
]

DOB_PATTERNS = [
    r"\b(\d{2}-\d{2}-\d{4})\b",
    r"\b(\d{4}-\d{2}-\d{2})\b",
]

IBAN_PATTERN = r"\bNL\d{2}[A-Z]{4}\d{10}\b"


def _regex_extract_basic_entities(text: str) -> dict:
    out = {
        "license_plates": [],
        "phone_numbers": [],
        "dob_candidates": [],
        "ibans": [],
    }
    for pat in LICENSE_PLATE_PATTERNS:
        out["license_plates"] += re.findall(pat, text)
    for pat in PHONE_PATTERNS:
        out["phone_numbers"] += re.findall(pat, text)
    for pat in DOB_PATTERNS:
        out["dob_candidates"] += re.findall(pat, text)
    out["ibans"] += re.findall(IBAN_PATTERN, text)
    # dedupe
    for k in out:
        seen = set()
        dedup = []
        for v in out[k]:
            if v not in seen:
                seen.add(v)
                dedup.append(v)
        out[k] = dedup
    return out


VOLATILE_PROMPT_HEADER = (
    "Je krijgt een tekst uit een politiedossier met paginamarkers [Pagina N].\n"
    "Extraheer uitsluitend wat letterlijk in de tekst staat en geef gestructureerde JSON terug.\n"
    "Gebruik géén interpretatie of afleiding; alleen tekst die aanwezig is.\n\n"
    "Output: één JSON-object, geen extra tekst. Velden:\n"
    "- persons: [{name, role (verdachte|getuige|slachtoffer|overig), dob (YYYY-MM-DD of null), aliases:[], doc: string, page: number|null}]\n"
    "- locations: [{name, type (adres|plaats|object|null), doc, page}]\n"
    "- vehicles: [{license_plate, brand: string|null, model: string|null, doc, page}]\n"
    "- phones: [{number, owner: string|null, doc, page}]\n"
    "- offenses: [{label, pack_id: string|null, doc, evidence_pages: [numbers]}]\n"
    "- organizations: [{name, type: string|null, doc, page}]\n"
    "Regels:\n"
    "- Neem waar mogelijk paginanummers over via [Pagina N].\n"
    "- Laat velden weg als er niets over te vinden is.\n"
)


def _build_volatile_prompt(pdf_title: str, pdf_text_chunk: str) -> str:
    return (
        f"DOCUMENT: {pdf_title}\n\n" +
        VOLATILE_PROMPT_HEADER +
        "\nTEKST START\n" + pdf_text_chunk + "\nTEKST EINDE"
    )


def _safe_json_parse(s: str) -> dict:
    try:
        obj = json.loads(s.strip())
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}


def _merge_volatile(into: dict, add: dict) -> dict:
    for key in ["persons", "locations", "vehicles", "phones", "offenses", "organizations"]:
        base = into.setdefault(key, [])
        items = add.get(key) or []
        for it in items:
            t = json.dumps(it, sort_keys=True, ensure_ascii=False)
            if not any(json.dumps(x, sort_keys=True, ensure_ascii=False) == t for x in base):
                base.append(it)
    for key in ["license_plates", "phone_numbers", "dob_candidates", "ibans"]:
        if add.get(key):
            base = into.setdefault(key, [])
            for v in add[key]:
                if v not in base:
                    base.append(v)
    return into


def _chunk_by_chars(text: str, max_chars: int = 12000) -> list:
    chunks = []
    current = []
    total = 0
    for line in text.splitlines(True):
        ln = len(line)
        if total + ln > max_chars and current:
            chunks.append("".join(current))
            current = []
            total = 0
        current.append(line)
        total += ln
    if current:
        chunks.append("".join(current))
    return chunks


def extract_volatile_context_for_doc(pdf_name: str, pdf_text: str) -> dict:
    result = {}
    regex_baseline = _regex_extract_basic_entities(pdf_text)
    result = _merge_volatile(result, regex_baseline)
    for chunk in _chunk_by_chars(pdf_text, max_chars=12000):
        prompt = _build_volatile_prompt(pdf_name, chunk)
        raw = post_chat([{"role": "user", "content": prompt}], timeout=180, stream=False)
        parsed = _safe_json_parse(raw)
        if parsed:
            result = _merge_volatile(result, parsed)
    return result


def write_volatile_json(target_path: str, data: dict) -> None:
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def secure_delete(path: str, passes: int = 1) -> None:
    try:
        if not os.path.exists(path):
            return
        size = os.path.getsize(path)
        with open(path, "r+b") as f:
            for _ in range(max(1, passes)):
                f.seek(0)
                f.write(b"\x00" * size)
                f.flush()
                os.fsync(f.fileno())
        os.remove(path)
    except Exception:
        try:
            os.remove(path)
        except Exception:
            pass


def build_hints_from_volatile(data: dict, max_terms: int = 50) -> list:
    terms = []
    for p in data.get("persons", []):
        if p.get("name"):
            terms.append(str(p["name"]))
        for a in p.get("aliases", []) or []:
            terms.append(str(a))
    for l in data.get("locations", []):
        if l.get("name"):
            terms.append(str(l["name"]))
    for v in data.get("vehicles", []):
        if v.get("license_plate"):
            terms.append(str(v["license_plate"]))
        if v.get("brand"):
            terms.append(str(v["brand"]))
        if v.get("model"):
            terms.append(str(v["model"]))
    for ph in data.get("phones", []):
        if ph.get("number"):
            terms.append(str(ph["number"]))
    for off in data.get("offenses", []):
        if off.get("label"):
            terms.append(str(off["label"]))
        if off.get("pack_id"):
            terms.append(str(off["pack_id"]))
    for raw in data.get("license_plates", []):
        terms.append(str(raw))
    for raw in data.get("phone_numbers", []):
        terms.append(str(raw))
    seen = set()
    out = []
    for t in terms:
        t2 = t.strip()
        if not t2:
            continue
        low = t2.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(t2)
        if len(out) >= max_terms:
            break
    return out

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

def build_doc_prompt(pdf_name: str, user_query: str, pdf_text: str, hints: list | None = None) -> str:
    hints_block = ""
    if hints:
        hints_block = "\nContext-hints (uitsluitend gebruiken om relevante citaten te vinden; geen extra interpretatie):\n- " + "\n- ".join(hints[:80]) + "\n"
    return (
        f"Document: {pdf_name}\n\n"
        f"{BASE_PROMPT}\n"
        f'Zoekopdracht: "{user_query}"\n'
        f"{hints_block}\n"
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

def _build_page_index(doc_text: str) -> list:
    idx = []
    for m in re.finditer(r"\[Pagina\s*(\d+)\]", doc_text):
        try:
            p = int(m.group(1))
        except Exception:
            p = None
        idx.append((m.start(), p))
    return idx

def _page_for_pos(pos: int, page_idx: list) -> int | None:
    if not page_idx:
        return None
    last = None
    for i, p in page_idx:
        if i <= pos:
            last = p
        else:
            break
    return last

def _flex_regex_from_quote(quote: str) -> str:
    q = quote.strip().strip('"“”\'\'')
    # escape then allow flexible whitespace
    q = re.escape(q)
    q = re.sub(r"\\\s+", "\\s+", q)  # normalize any escaped whitespace groups
    q = re.sub(r"\s+", "\\s+", q)
    return q

def enforce_exact_quotes_for_doc(parsed: list, doc_text: str, pdf_name: str) -> list:
    page_idx = _build_page_index(doc_text)
    verified = []
    for it in parsed:
        citaat = (it.get("citaat") or "").strip()
        if not citaat:
            continue
        # build whitespace-flexible regex
        pattern = _flex_regex_from_quote(citaat)
        m = re.search(pattern, doc_text, flags=re.S)
        if not m:
            # try without ellipses if any
            c2 = citaat.replace("…", " ").replace("...", " ")
            pattern2 = _flex_regex_from_quote(c2)
            m = re.search(pattern2, doc_text, flags=re.S)
        if not m:
            # reject unverifiable quote
            continue
        exact = doc_text[m.start():m.end()]
        pagina = it.get("pagina")
        if pagina is None:
            pagina = _page_for_pos(m.start(), page_idx)
        verified.append({
            "citaat": exact,
            "pagina": pagina,
            "titel": it.get("titel") or pdf_name,
            "document": pdf_name,
        })
    return verified

def extract_citaten_from_dir(pdf_dir: str, vraag: str, hints: list | None = None) -> list:
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
        prompt = build_doc_prompt(os.path.basename(pdf), vraag, text, hints=hints)
        try:
            raw = post_chat([{"role": "user", "content": prompt}], timeout=240, stream=False)
        except requests.RequestException as e:
            print(f"[!] API-fout bij {pdf}: {e}")
            continue
        parsed = parse_output(raw, pdf)
        # Enforce: only accept quotes that are exact substrings of the source text
        verified = enforce_exact_quotes_for_doc(parsed, text, pdf)
        alle.extend(verified)
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
    # 0) Context packs activeren
    ctx_dir = os.getenv("CONTEXT_DIR", os.path.join(os.getcwd(), "context"))
    corpus_sample = get_corpus_sample_text(pdf_dir, max_pages=2, max_chars=30000)
    common, active_packs = load_context_packs(ctx_dir, vraag, corpus_sample)
    hints = build_context_hints(active_packs) if active_packs else []

    # 0b) Volatile context extractie en opslag
    volatile_path = os.path.join(pdf_dir, ".volatile_context.json")
    aggregate = {}
    try:
        pdfs = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
        for pdf in pdfs:
            path = os.path.join(pdf_dir, pdf)
            try:
                text = extract_text_from_pdf(path)
            except Exception as e:
                print(f"[!] Kon {pdf} niet lezen voor context-extractie: {e}")
                continue
            per_doc = extract_volatile_context_for_doc(os.path.basename(pdf), text)
            aggregate = _merge_volatile(aggregate, per_doc)
        write_volatile_json(volatile_path, aggregate)
    except Exception as e:
        print(f"[!] Context-extractie fout: {e}")

    # 0c) Hints uitbreiden met volatile entiteiten
    if aggregate:
        hints_vol = build_hints_from_volatile(aggregate)
        seen = set()
        merged = []
        for h in list(hints) + hints_vol:
            k = h.lower()
            if k not in seen:
                seen.add(k)
                merged.append(h)
        hints = merged

    # 1) Alle citaten
    alle_citaten = extract_citaten_from_dir(pdf_dir, vraag, hints=hints)

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

    # 6) Volatile JSON veilig verwijderen i.v.m. WPG
    try:
        secure_delete(volatile_path, passes=1)
    except Exception:
        pass

if __name__ == "__main__":
    # Pas deze twee aan
    PDF_DIR = os.getenv("PDF_DIR", "/Users/116299/Documents/Project_Afstuderen/test")
    USER_QUERY = os.getenv("USER_QUERY", "Welke plaatsnamen worden genoemd en corresponderen zij met strafbare feiten?")
    try:
        main(PDF_DIR, USER_QUERY)
    except KeyboardInterrupt:
        print("\n[!] Afgebroken door gebruiker.")

