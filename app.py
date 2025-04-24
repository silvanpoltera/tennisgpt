from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

load_dotenv()
app = Flask(__name__)

# Lade Dokumente
loader = TextLoader("Wissen/swiss-tennis-gpt.txt", encoding='utf8')
documents = loader.load()

# Splitten
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(documents)

# Vektorstore
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embedding)

# Chain
chain = load_qa_chain(ChatOpenAI(model="gpt-4", temperature=0), chain_type="stuff")

@app.route("/antwort", methods=["POST"])
def antwort():
    frage = request.json.get("frage")
    docs = vectorstore.similarity_search(frage)
    response = chain.run(input_documents=docs, question=frage)
    return jsonify({"antwort": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
