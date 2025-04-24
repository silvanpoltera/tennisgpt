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
    docs = vectorstore.similarity_search(frage)
    response = chain.run(input_documents=docs, question=frage)
    return jsonify({"antwort": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
