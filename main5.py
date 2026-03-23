import chromadb
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Summary için gerekli yeni sınıflar
from langchain.memory import ConversationSummaryMemory

# --- AYARLAR ---
PDF_PATH = "tck.pdf"
COLLECTION_NAME = "pdf_bilgileri"

# 1. PDF ve ChromaDB Hazırlığı (Aynı kalıyor)
loader = PyPDFLoader(PDF_PATH)
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(data)
documents = [doc.page_content for doc in chunks]
ids = [f"chunk_{i}" for i in range(len(documents))]

client = chromadb.HttpClient(host='localhost', port=8000)
collection = client.get_or_create_collection(name=COLLECTION_NAME)
collection.upsert(documents=documents, ids=ids)

# 2. LLM ve Summary Memory Kurulumu
llm = OllamaLLM(model="llama3")

# Bellek burada tanımlanıyor. 
# llm=llm parametresi önemli; çünkü özeti bu model çıkaracak.
memory = ConversationSummaryMemory(llm=llm, memory_key="chat_summary")

# 3. Sorgu Döngüsü
queries = ["Hırsızlık suçu nedir?", "Cezası ne kadar?", "Gece işlenirse ne olur?"]

for query in queries:
    print(f"\n🔍 Soru: {query}")
    
    # RAG: Veri çekme
    results = collection.query(query_texts=[query], n_results=2)
    context = " ".join(results['documents'][0])

    # ÖZETİ GETİR: Şu ana kadarki konuşmanın özeti
    summary = memory.load_memory_variables({})["chat_summary"]

    prompt = f"""
    Sen hukuk asistanısın. 
    ÖNCEKİ KONUŞMALARIN ÖZETİ: {summary}
    BAĞLAM (PDF): {context}
    SORU: {query}
    CEVAP:"""

    response = llm.invoke(prompt)
    print(f"🤖 Cevap: {response}")

    # BELLEĞE KAYDET: Bu adımda arka planda LLM çalışıp özeti günceller
    memory.save_context({"input": query}, {"output": response})