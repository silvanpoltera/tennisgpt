from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredImageLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()
app = Flask(__name__)

# Der Pfad zum Ordner, in dem sich die Wissensdateien befinden
wissen_ordner = "Wissen"

# Erstelle eine leere Liste für alle Dokumente
documents = []

# Gehe durch alle Dateien im Wissen-Ordner und lade sie abhängig vom Typ
for filename in os.listdir(wissen_ordner):
    file_path = os.path.join(wissen_ordner, filename)

    if filename.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        pdf_docs = loader.load()
        documents.extend(pdf_docs)

    elif filename.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
        txt_docs = loader.load()
        documents.extend(txt_docs)

    elif filename.endswith(".png"):
        loader = UnstructuredImageLoader(file_path)
        img_docs = loader.load()
        documents.extend(img_docs)

# Dokumente splitten
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(documents)

# Vektorstore erstellen
embedding = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embedding)

# LLMChain mit fester Systemanweisung
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

    # Antwort zusammenbauen mit HTML
    antwort_text = (
        "<b>Hey lieber Tennis Fan,</b><br><br>"
        f"{response}<br><br>"
        f"{quellen_text}<br><br>"
        "Wir hoffen, dass wir dir mit dieser Antwort helfen konnten.<br>"
        "Falls die Informationen nicht ausreichend waren, wende dich bitte an eure interne IT-Abteilung,<br>"
        "damit diese ein Ticket in Click Up erstellen kann.<br><br>"
        "Wir danken dir und wünschen dir noch einen schönen Tag.<br><br>"
        "Liebe Gruesse<br>"
        "Dein Swiss Tennis CMS Support Team"
    )

    return jsonify({"antwort": antwort_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
