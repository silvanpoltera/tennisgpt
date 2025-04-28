from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

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

# Chain für die Frage-Antwort-Suche
chain = load_qa_chain(ChatOpenAI(model="gpt-4", temperature=0), chain_type="stuff")

@app.route("/antwort", methods=["POST"])
def antwort():
    frage = request.json.get("frage")

    # Hinweis an das Modell hinzufügen, in Du-Form zu antworten
    angepasste_frage = (
        "Bitte beantworte die folgende Frage in der Du-Form und sehr freundlich formuliert:\n"
        f"{frage}"
    )

    docs = vectorstore.similarity_search(frage)
    response = chain.run(input_documents=docs, question=angepasste_frage)

    # Formatierte Antwort erstellen
    antwort_text = (
        "Hey lieber Tennis Fan,\n\n"
        "------------------------------\n\n"
        f"{response}\n\n"
        "------------------------------\n\n"
        "Wir hoffen, dass wir dir mit dieser Antwort helfen konnten.\n"
        "Falls die Informationen nicht ausreichend waren, wende dich bitte an eure interne IT-Abteilung,\n"
        "damit diese ein Ticket in Click Up erstellen kann.\n\n"
        "Wir danken dir und wünschen dir noch einen schönen Tag.\n\n"
        "Liebe Grüße\n"
        "Dein Swiss Tennis CMS Support Team"
    )

    return jsonify({"antwort": antwort_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
