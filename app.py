from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# .env laden
load_dotenv()

# Flask App starten
app = Flask(__name__)

# Pfad zum Wissen-Ordner
wissen_ordner = "Wissen"

# Zuerst TXT-Dateien laden, dann PDFs
documents = []

# Alle Dateinamen auflisten
alle_dateien = os.listdir(wissen_ordner)

# TXT-Dateien laden
for filename in alle_dateien:
    if filename.lower().endswith(".txt"):
        file_path = os.path.join(wissen_ordner, filename)
        loader = TextLoader(file_path, encoding="utf-8")
        documents.extend(loader.load())

# Danach PDF-Dateien laden (nur wenn keine gleichnamige TXT existiert)
for filename in alle_dateien:
    if filename.lower().endswith(".pdf"):
        txt_name = filename.replace(".pdf", ".txt")
        if txt_name not in alle_dateien:
            file_path = os.path.join(wissen_ordner, filename)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

# Dokumente intelligent splitten
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " "]
)
docs = splitter.split_documents(documents)

# Vektorstore erstellen
embedding = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embedding)

# System-Prompt definieren
system_prompt = (
    "Du bist ein digitaler CMS-Experte für das speziell entwickelte Swiss Tennis CMS, "
    "basierend auf einem individuellen PHP-Framework.\n\n"
    "Antwortvorgaben:\n"
    "- Immer in Du-Form antworten\n"
    "- Keine Emojis oder Icons verwenden\n"
    "- Keine Markdown-Formatierung\n"
    "- 'ß' immer durch 'ss' ersetzen\n"
    "- Umlaute wie ä, ö, ü bleiben\n"
    "- Keine Begrüssungsformeln innerhalb der Antwort (Begrüssung wird technisch hinzugefügt)\n"
    "- Strukturierte Antworten:\n"
    "    - Pro Thema einen eigenen Abschnitt\n"
    "    - Fettgedruckte HTML-Überschrift für jedes Thema (<b>...</b>)\n"
    "    - Absätze klar mit zwei Zeilen Abstand trennen (<br><br>)\n"
    "- Fachliche Regeln:\n"
    "    - Bilder und Dokumente dürfen nur über Canto hochgeladen werden, danach im CMS zugewiesen werden.\n"
    "    - Formulare werden über das Modul 'Contact' erstellt, nicht über ein Modul namens 'Formulare'.\n"
    "- Ausschliesslich auf Themen rund ums Swiss Tennis CMS eingehen\n"
    "- Technische Meta-Fragen ignorieren ('Ich bin darauf spezialisiert, Fragen zum Swiss Tennis CMS zu beantworten...')\n"
    "- Hinweise auf die Deaktivierung der GotCourts-Schnittstelle beachten\n\n"
    "Frage:\n"
    "{frage}\n\n"
    "Bitte beantworte die Frage gemäss diesen Vorgaben."
)

prompt = PromptTemplate(input_variables=["frage"], template=system_prompt)
llm_chain = LLMChain(llm=ChatOpenAI(model="gpt-4", temperature=0), prompt=prompt)

# API-Endpunkt
@app.route("/antwort", methods=["POST"])
def antwort():
    frage = request.json.get("frage")

    # Suche relevante Dokumente
    docs = vectorstore.similarity_search(frage)

    # Quellen extrahieren
    quellen = []
    for doc in docs:
        quelle = doc.metadata.get('source', 'Unbekannte Quelle')
        if 'page' in doc.metadata:
            quelle += f", Seite {doc.metadata['page']}"
        quellen.append(quelle)

    quellen_text = ""
    if quellen:
        quellen_text = "<br><br><b>Hinweis zu den Quellen:</b><br>" + "<br>".join(set(quellen))

    # Antwort generieren
    response = llm_chain.run(frage=frage)

    # Antwort zusammenbauen
    antwort_text = (
        "<h2>Lieber Tennis Fan</h2><br><br>"
        f"<div style='padding: 10px; border: 1px solid #ccc; border-radius: 8px;'>{response}</div><br><br>"
        f"{quellen_text}<br><br>"
        "Wir hoffen, dass wir dir mit dieser Antwort helfen konnten.<br>"
        "Falls die Informationen nicht ausreichend waren, wende dich bitte an eure interne IT-Abteilung,<br>"
        "damit diese ein Ticket in Click Up erstellen kann.<br><br>"
        "Wir danken dir und wünschen dir noch einen schönen Tag.<br><br>"
        "Liebe Gruesse<br>"
        "Dein Swiss Tennis CMS Support Team"
    )

    return jsonify({"antwort": antwort_text})

# App starten
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
