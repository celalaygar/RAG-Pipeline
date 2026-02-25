import chromadb
from langchain_ollama import OllamaLLM

# 1. Docker üzerindeki ChromaDB'ye bağlan
try:
    client = chromadb.HttpClient(host='localhost', port=8000)
    # Koleksiyonu al veya yoksa oluştur
    collection = client.get_or_create_collection(name="sirket_bilgileri")
    print("✅ ChromaDB bağlantısı başarılı.")
except Exception as e:
    print(f"❌ ChromaDB bağlantı hatası: {e}\nLütfen Docker container'ın çalıştığından emin olun.")
    exit()

# 2. Verileri Hazırla ve Ekle
# Not: Aynı ID ile tekrar ekleme yapmamak için kontrol ekleyebilirsiniz 
# veya her seferinde baştan eklemek yerine mevcut veriyi kullanabilirsiniz.
documents = [
    "Şirketimizde uzaktan çalışma politikası Pazartesi ve Cuma günlerini kapsar.",
    "Yıllık izin hakkı, kıdemi 1-5 yıl arası olan çalışanlar için 14 iş günüdür.",
    "Yıllık izin hakkı, kıdemi 5-10 yıl arası olan çalışanlar için 20 iş günüdür.",
    "Yıllık izin hakkı, kıdemi 10+ yıl arası olan çalışanlar için 26 iş günüdür.",
    "Yemek kartı bakiyeleri her ayın 1'inde yüklenmektedir.",
    "Çalışanlarımızın sağlık sigortası kapsamı, temel ve ek teminatları içermektedir.",
    "Çalışanlarımızın yıllık performans değerlendirmeleri, her yılın Aralık ayında yapılır.",
    "Çalışanlarımızın yıllık izin hakları, kıdemlerine göre değişiklik göstermektedir. Kıdemi 1-5 yıl arasında olan çalışanlar 14 iş günü, 5-10 yıl arasında olanlar 20 iş günü, 10 yıldan fazla olanlar ise 26 iş günü yıllık izne sahiptir.",
    "Çalışanlarımızın çalışma saatleri, haftalık 40 saat olarak belirlenmiştir. Çalışanlar, haftanın 5 günü, günde 8 saat çalışmaktadır.",
    "Çalışanlarımızın sağlık sigortası kapsamı temel ve ek teminatları içerir: Hastane, doktor ve ilaç (temel); diş, göz ve alternatif tıp (ek)."
]
ids = [f"id{i+1}" for i in range(len(documents))]

# Verileri ekleyelim (upsert kullanarak varsa günceller, yoksa ekler)
collection.upsert(
    documents=documents,
    ids=ids
)
print(f"✅ {len(documents)} döküman veritabanına işlendi.")

# 3. Sorgulama (Retrieval)
query = "12 yıldır bu şirketteyim, kaç gün izin hakkım var?"
print(f"\n🔍 Sorgulanıyor: {query}")

results = collection.query(
    query_texts=[query],
    n_results=2  # En alakalı 2 parçayı getir ki model daha iyi analiz etsin
)

# Bulunan parçaları birleştirelim
context = " ".join(results['documents'][0])

# 4. Üretim (Generation) - Ollama Entegrasyonu
try:
    llm = OllamaLLM(model="llama3")
    
    prompt = f"""
    Sen bir şirket asistanısın. Aşağıdaki bağlam (context) bilgilerini kullanarak kullanıcıya nazikçe cevap ver.
    Cevabı sadece verilen bilgilere dayandır. Bilgi dışına çıkma.
    
    BAĞLAM:
    {context}
    
    SORU:
    {query}
    
    CEVAP:
    """
    
    print("🤖 Yapay zeka yanıt üretiyor...\n")
    response = llm.invoke(prompt)
    
    print("-" * 40)
    print(f"KAYNAK BİLGİ: {context}")
    print("-" * 40)
    print(f"YAPAY ZEKA YANITI:\n{response}")
    print("-" * 40)

except Exception as e:
    print(f"❌ Ollama hatası: {e}\nLütfen Ollama'nın açık ve llama3 modelinin yüklü olduğundan emin olun.")