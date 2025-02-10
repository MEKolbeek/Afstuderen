import os
import pandas as pd
import requests
import json
import unicodedata

# Configuratie voor LLaMA API
API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"

# Functie om citaten te normaliseren
def preprocess_citation(citation):
    if isinstance(citation, str):
        citation = unicodedata.normalize('NFKD', citation).encode('ASCII', 'ignore').decode('utf-8')
        citation = citation.replace("\n", " ").strip()
        return citation
    return "Geen citaat beschikbaar"

# Functie om een prompt naar LLaMA te sturen en de response te verwerken
def query_llama(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": 50,
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=60, stream=True)
        response.raise_for_status()

        # Verwerk de streaming response
        lines = response.iter_lines(decode_unicode=True)
        full_text = ""
        for line in lines:
            if line.strip():  # Controleer op lege regels
                try:
                    json_line = json.loads(line)
                    if "response" in json_line:
                        full_text += json_line["response"]
                except json.JSONDecodeError:
                    print(f"Fout bij verwerken van regel: {line}")

        return full_text.strip()
    except requests.exceptions.RequestException as e:
        return f"Fout: {e}"
    except json.JSONDecodeError:
        return "JSON decode error: Ongeldige serverrespons"

# Functie om kernwoorden uit citaten te halen en correct te splitsen
def generate_keywords_from_citation(citation):
    if isinstance(citation, str):
        prompt = (
            f"Hier is een citaat:\n\n{citation}\n\n"
            "Kies 1, 2 of 3 kernwoorden die op basis van de context van de citaat deze het beste omschrijven. "
            "Geef de kernwoorden terug als een lijst gescheiden door komma's, zonder extra tekst."
        )
        response = query_llama(prompt)

        # Splits de kernwoorden correct op komma's en verwijder eventuele extra spaties
        keywords = [kw.strip() for kw in response.split(",") if kw.strip()]
        return keywords[:3]  # Maximaal 3 kernwoorden
    return ["", "", ""]  # Lege waarden voor het geval er geen kernwoorden zijn

# Start van het script
if __name__ == "__main__":
    input_file = r"C:\Users\mkolb\Downloads\PESTEL-MASTER-normtest.xlsx"
    output_directory = r"C:\Users\mkolb\Downloads"
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Het bestand '{input_file}' is niet gevonden.")

    data = pd.read_excel(input_file)

    if "Citaat" not in data.columns:
        raise ValueError("De kolom 'Citaat' ontbreekt in het bestand.")

    # Normaliseer de citaten
    data["Genormaliseerd Citaat"] = data["Citaat"].apply(preprocess_citation)

    print("Start met het genereren van kernwoorden...")
    kernwoorden_1, kernwoorden_2, kernwoorden_3 = [], [], []

    for index, row in data.iterrows():
        print(f"Bezig met citaat {index + 1}: {row['Genormaliseerd Citaat']}")
        keywords = generate_keywords_from_citation(row["Genormaliseerd Citaat"])
        
        # Vul de kolommen correct met kernwoorden
        kernwoorden_1.append(keywords[0] if len(keywords) > 0 else "")
        kernwoorden_2.append(keywords[1] if len(keywords) > 1 else "")
        kernwoorden_3.append(keywords[2] if len(keywords) > 2 else "")

    # Voeg de kernwoorden toe aan de dataframe
    data["Kernwoorden_1"] = kernwoorden_1
    data["Kernwoorden_2"] = kernwoorden_2
    data["Kernwoorden_3"] = kernwoorden_3

    # Sla de resultaten op als CSV met unieke naam
    output_file = os.path.join(output_directory, "PESTEL-MASTER-KERNWOORDEN.csv")
    counter = 1
    while os.path.exists(output_file):
        output_file = os.path.join(output_directory, f"PESTEL-MASTER-KERNWOORDEN_{counter}.csv")
        counter += 1

    data.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Resultaten opgeslagen in: {output_file}")
