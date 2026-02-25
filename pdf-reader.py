import chromadb
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- AYARLAR ---
PDF_PATH = "tck.pdf"  # PDF dosyanızın adı
COLLECTION_NAME = "pdf_bilgileri"

# 1. PDF Yükleme ve Parçalama (Chunking)
print(f"📄 {PDF_PATH} okunuyor...")
loader = PyPDFLoader(PDF_PATH)
data = loader.load()

# PDF'i küçük parçalara bölelim (Modelin her parçayı daha iyi anlaması için)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(data)

# Parçaları ChromaDB'nin anlayacağı formata sokalım
documents = [doc.page_content for doc in chunks]
ids = [f"chunk_{i}" for i in range(len(documents))]

# 2. ChromaDB Bağlantısı
try:
    client = chromadb.HttpClient(host='localhost', port=8000)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    
    # PDF verilerini ekleyelim
    collection.upsert(documents=documents, ids=ids)
    print(f"✅ {len(documents)} parça veritabanına yüklendi.")
except Exception as e:
    print(f"❌ Bağlantı hatası: {e}")
    exit()

# 3. Sorgulama ve Üretim
query = "İştirak halinde işlenen suçlarda, sadece gönüllü vazgeçen suç ortağı neyden yararlanır?"
print(f"\n🔍 Sorgulanıyor: {query}")

results = collection.query(query_texts=[query], n_results=3)
context = " ".join(results['documents'][0])



try:
    llm = OllamaLLM(model="llama3")
    
    prompt = f"""
    Sen dökümanları analiz eden bir asistansın. 
    Sadece aşağıdaki BAĞLAM bilgilerini kullanarak soruyu cevapla.
    Cevabı verirken nazik ol ve dökümanın dışına çıkma.

    BAĞLAM:
    {context}

    SORU:
    {query}

    CEVAP:
    """
    
    print("🤖 Yanıt oluşturuluyor...")
    response = llm.invoke(prompt)
    
    print("\n" + "="*50)
    print(f"PDF'den Alınan Bilgi Özeti: {context[:200]}...")
    print("="*50)
    print(f"YAPAY ZEKA YANITI:\n{response}")
    print("="*50)

except Exception as e:
    print(f"❌ Ollama hatası: {e}")