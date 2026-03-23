import chromadb
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory

# --- AYARLAR ---
PDF_PATH = "tck.pdf"
COLLECTION_NAME = "pdf_bilgileri"

# 1. PDF Yükleme ve Parçalama
try:
    print(f"📄 {PDF_PATH} okunuyor...")
    loader = PyPDFLoader(PDF_PATH)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    documents = [doc.page_content for doc in chunks]
    ids = [f"chunk_{i}" for i in range(len(documents))]
except Exception as e:
    print(f"❌ PDF Hatası: {e}")
    exit()

# 2. ChromaDB Bağlantısı
try:
    client = chromadb.HttpClient(host='localhost', port=8000)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    collection.upsert(documents=documents, ids=ids)
    print(f"✅ {len(documents)} parça veritabanına yüklendi.")
except Exception as e:
    print(f"❌ ChromaDB bağlantı hatası: {e}")
    exit()

# 3. Modern Bellek (History) ve LLM Kurulumu
history = ChatMessageHistory() # Geçmişi burada tutacağız

try:
    llm = OllamaLLM(model="llama3")
except Exception as e:
    print(f"❌ Ollama hatası: {e}")
    exit()

# 4. Sohbet Döngüsü
queries = [
    "İştirak halinde işlenen suçlarda, sadece gönüllü vazgeçen suç ortağı neyden yararlanır?",
    "TCK'ya göre hırsızlık suçunun cezası nedir?",
    "Peki, bu suç çocuklara karşı işlenirse ceza artar mı?" # Hafızayı test eden soru
]

print("\n🤖 TCK Asistanı Hazır!")



for i, query in enumerate(queries):
    print("\n" + "="*40)
    print(f"SORU {i+1}: {query}")

    # RAG: Veritabanından döküman çekme
    results = collection.query(query_texts=[query], n_results=3)
    context = " ".join(results['documents'][0])

    # Geçmiş mesajları metin olarak birleştirme
    past_messages = ""
    for msg in history.messages[-4:]: # Sadece son 4 mesajı al (Bellek yönetimi)
        prefix = "Kullanıcı" if msg.type == "human" else "Asistan"
        past_messages += f"{prefix}: {msg.content}\n"

    prompt = f"""
    Sen uzman bir hukuk asistanısın. Aşağıdaki bağlamı ve konuşma geçmişini kullanarak soruyu cevapla.
    
    KONUŞMA GEÇMİŞİ:
    {past_messages}
    
    BAĞLAM (KANUN METNİ):
    {context}
    
    SORU: 
    {query}
    
    CEVAP:"""

    try:
        print("🤖 Cevap üretiliyor...")
        response = llm.invoke(prompt)
        print(f"\nASİSTAN: {response}")
        
        # Mesajları geçmişe ekle
        history.add_user_message(query)
        history.add_ai_message(response)
        
    except Exception as e:
        print(f"❌ Hata: {e}")