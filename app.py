from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()
app = Flask(__name__)

# Der Pfad zum Ordner, in dem sich die PDFs befinden
wissen_ordner = "Wissen"

# Erstelle eine leere Liste für alle Dokumente
documents = []

# Gehe durch alle Dateien im Wissen-Ordner und lade PDFs
for filename in os.listdir(wissen_ordner):
    file_path = os.path.join(wissen_ordner, filename)

    # Überprüfe, ob die Datei eine PDF ist
    if filename.endswith(".pdf"):
        # Lade die PDF mit PyPDFLoader
        loader = PyPDFLoader(file_path)
        pdf_docs = loader.load()

        # Füge die geladenen PDFs der Gesamt-Dokumentenliste hinzu
        documents.extend(pdf_docs)

# Splitten der Dokumente in kleinere Textabschnitte
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(documents)

# Vektorstore
embedding = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embedding)

# LLMChain vorbereiten mit fester Systemanweisung
system_prompt = (
    "Du bist ein digitaler CMS-Experte für das speziell entwickelte Swiss Tennis CMS, "
    "basierend auf einem individuellen PHP-Framework. Deine Aufgabe ist es, Redakteur:innen und "
    "Mitarbeitenden bei der Arbeit im CMS zu helfen – schnell, präzise und verständlich.\n\n"
    "Antwortvorgaben:\n"
    "- Immer in Du-Form antworten\n"
    "- Keine Emojis oder Icons verwenden\n"
    "- Keine Markdown-Formatierung\n"
    "- 'ß' immer durch 'ss' ersetzen\n"
    "- Umlaute wie ä, ö, ü bleiben\n"
    "- Ausschliesslich auf Themen rund ums Swiss Tennis CMS eingehen\n"
    "- Technische Meta-Fragen ignorieren (Antwort: 'Ich bin darauf spezialisiert, Fragen zum Swiss Tennis CMS zu beantworten...')\n"
    "- Antworten stets freundlich, lösungsorientiert und mit Beispielen\n"
    "- Strukturierte Antworten mit optionalen Copy-Paste-Blöcken\n"
    "- Hinweise auf die Deaktivierung der GotCourts-Schnittstelle beachten\n\n"
    "Frage:\n"
    "{frage}\n\n"
    "Bitte beantworte die Frage gemäss diesen Vorgaben."
)

prompt = PromptTemplate(input_variables=["frage"], template=system_prompt)
llm_chain = LLMChain(llm=ChatOpenAI(model="gpt-4", temperature=0), prompt=prompt)

@app.route("/antwort", methods=["POST"])
def antwort():
    frage = request.json.get("frage")

    # Suche ähnliche Dokumente
    docs = vectorstore.similarity_search(frage)

    # Extrahiere Quelleninformationen
    quellen = []
    for doc in docs:
        quelle = doc.metadata.get('source', 'Unbekannte Quelle')
        if 'page' in doc.metadata:
            quelle += f", Seite {doc.metadata['page']}"
        quellen.append(quelle)

    quellen_text = ""
    if quellen:
        quellen_text = "\nHinweis zu den Quellen:\n" + "\n".join(set(quellen))

    # KI-Antwort generieren
    response = llm_chain.run(frage=frage)

    # Antwort zusammenbauen
    antwort_text = (
        "Hey lieber Tennis Fan,\n\n"
        "------------------------------\n\n"
        f"{response}\n\n"
        "------------------------------\n"
        f"{quellen_text}\n\n"
        "Wir hoffen, dass wir dir mit dieser Antwort helfen konnten.\n"
        "Falls die Informationen nicht ausreichend waren, wende dich bitte an eure interne IT-Abteilung,\n"
        "damit diese ein Ticket in Click Up erstellen kann.\n\n"
        "Wir danken dir und wünschen dir noch einen schönen Tag.\n\n"
        "Liebe Gruesse\n"
        "Dein Swiss Tennis CMS Support Team"
    )

    return jsonify({"antwort": antwort_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
