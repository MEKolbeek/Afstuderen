import pandas as pd
import unicodedata

# Pad naar het originele Excel-bestand
input_file = "PESTEL-MASTER.xlsx"  # Vervang door het juiste pad
output_file = "PESTEL-MASTER-NORMALIZED.xlsx"  # Genormaliseerde uitvoer

# Functie om tekst te normaliseren
def normalize_text(text):
    if isinstance(text, str):
        # Unicode normalisatie om speciale tekens te verwijderen
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        # Vervang newline-tekens en verwijder overbodige spaties
        text = text.replace("\n", " ").strip()
        # Houd alleen alfanumerieke tekens, spaties, enkele quotes, en streepjes over
        return ''.join(c for c in text if c.isalnum() or c.isspace() or c in ("'", "-"))
    return text

# Laad het Excel-bestand
try:
    data = pd.read_excel(input_file)
except FileNotFoundError:
    print(f"Bestand '{input_file}' niet gevonden. Zorg ervoor dat het bestand zich in de juiste map bevindt.")
    exit()

# Pas normalisatie toe op alle kolommen
for column in data.columns:
    data[column] = data[column].apply(normalize_text)

# Sla het genormaliseerde bestand op
data.to_excel(output_file, index=False)
print(f"Genormaliseerd bestand opgeslagen als: {output_file}")
