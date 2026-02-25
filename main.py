import chromadb

# 1. Docker'daki ChromaDB'ye bağlanalım
client = chromadb.HttpClient(host='localhost', port=8000)

# 2. Bir 'Koleksiyon' (Tablo gibi düşünebilirsiniz) oluşturalım
# ChromaDB varsayılan olarak "all-MiniLM-L6-v2" modelini kullanarak embedding yapar.
collection = client.get_or_create_collection(name="sirket_bilgileri")

# 3. Bilgileri Veritabanına Ekleyelim (Ingestion)
# Normalde burası PDF'den veya veritabanından gelir.
documents = [
    "Şirketimizde uzaktan çalışma politikası Pazartesi ve Cuma günlerini kapsar.",
    "Yıllık izin hakkı, kıdemi 1-5 yıl arası olan çalışanlar için 14 iş günüdür.",
    "Yemek kartı bakiyeleri her ayın 1'inde yüklenmektedir.",
    "Çalışanlarımızın sağlık sigortası kapsamı, temel ve ek teminatları içermektedir.",
    "Çalışanlarımızın yıllık performans değerlendirmeleri, her yılın Aralık ayında yapılır.",
    "Çalışanlarımızın yıllık izin hakları, kıdemlerine göre değişiklik göstermektedir. Kıdemi 1-5 yıl arasında olan çalışanlar 14 iş günü, 5-10 yıl arasında olanlar 20 iş günü, 10 yıldan fazla olanlar ise 26 iş günü yıllık izne sahiptir.",
    "Çalışanlarımızın çalışma saatleri, haftalık 40 saat olarak belirlenmiştir. Çalışanlar, haftanın 5 günü, günde 8 saat çalışmaktadır.",
    "Çalışanlarımızın sağlık sigortası kapsamı, temel ve ek teminatları içermektedir. Temel teminatlar arasında hastane masrafları, doktor ziyaretleri ve ilaç giderleri yer alırken, ek teminatlar arasında diş sağlığı, göz sağlığı ve alternatif tıp hizmetleri bulunmaktadır."
]
ids = ["id1", "id2", "id3", "id4", "id5", "id6", "id7", "id8"]

collection.add(
    documents=documents,
    ids=ids
)

# 4. Sorgulama (Retrieval)
query = "İzin haklarım ne kadar?"
results = collection.query(
    query_texts=[query],
    n_results=1 # En alakalı 1 dökümanı getir
)

relevant_doc = results['documents'][0][0]
print(f"Sorgu: {query}")
print(f"Bulunan En Alakalı Bilgi: {relevant_doc}")

# 5. Üretim (Generation) - Burası LLM'e gider
# prompt = f"Soru: {query}\n\nBilgi: {relevant_doc}\n\nBu bilgiye dayanarak cevap ver."
# response = llm.generate(prompt)
