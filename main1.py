import chromadb
# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM

# 1. Docker üzerindeki ChromaDB'ye bağlan
chroma_client = chromadb.HttpClient(host='localhost', port=8000)
collection = chroma_client.get_or_create_collection(name="bilgi_deposu")

# 2. Örnek Veri Ekleme (Daha önce eklediyseniz bu adımı geçebilirsiniz)
content = "Apple Silicon işlemcili Mac'lerde Docker performansı oldukça yüksektir."
collection.add(
    documents=[content],
    ids=["doc_01"]
)

# 3. Kullanıcı Sorgusu
user_query = "MacBook'larda Docker nasıl çalışır?"

# 4. RETRIEVAL (Bilgiyi Veritabanından Getirme)
results = collection.query(
    query_texts=[user_query],
    n_results=1
)
retrieved_info = results['documents'][0][0]

# 5. GENERATION (LLM ile Yanıt Üretme)
# llm = Ollama(model="llama3")
llm = OllamaLLM(model="llama3")

# Modelin kendi kafasından uydurmasını engelleyen "Prompt"
prompt = f"""
Sen yardımcı bir asistansın. Aşağıdaki bağlam bilgisine dayanarak soruyu cevapla.
Eğer bilgi bağlamda yoksa, bilmediğini söyle.

Bağlam: {retrieved_info}
Soru: {user_query}

Cevap:
"""

response = llm.invoke(prompt)

print("-" * 30)
print(f"Sistemden Gelen Bilgi: {retrieved_info}")
print(f"Yapay Zeka Yanıtı: {response}")