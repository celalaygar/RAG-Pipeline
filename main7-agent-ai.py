import os
from crewai import Agent, Task, Crew, Process, LLM
from pymongo import MongoClient

# 1. ORTAM VE LLM YAPILANDIRMASI
os.environ["OPENAI_API_KEY"] = "NA"
# CrewAI Memory için gerekli eksik ayar:
os.environ["EMBEDDINGS_OLLAMA_MODEL_NAME"] = "llama3" 

llama3 = LLM(
    model="ollama/llama3",
    base_url="http://127.0.0.1:11434"
)

# 2. MONGODB BAĞLANTISI (Kullanıcı adı ve şifre eklendi)
# NOT: Buradaki 'admin' ve 'password' kısımlarını kendi bilgilerine göre güncelle!

try:
    username = "root11"
    password = "371root1124"
    MONGO_URI = f"mongodb://{username}:{password}@localhost:27017/rag_pipeline?authSource=admin"
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    db = mongo_client["code_factory"]
    collection = db["saved_codes"]
    mongo_client.server_info() # Bağlantıyı ve yetkiyi doğrula
    print("✅ MongoDB bağlantısı başarılı.")
except Exception as e:
    print(f"❌ Uyarı: MongoDB bağlantısı kurulamadı: {e}")

# 3. ÖZEL ARAÇ: MongoDB'ye Kaydetme
def save_to_mongodb(code_content, task_name):
    try:
        data = {"task": task_name, "code": str(code_content), "status": "validated"}
        collection.insert_one(data)
        return "Kod başarıyla MongoDB'ye kaydedildi."
    except Exception as e:
        return f"MongoDB Kayıt Hatası: {e}"

# 4. AJANLARIN TANIMLANMASI
coder_agent = Agent(
    role='Kıdemli Python Geliştirici',
    goal='{topic} konusuna uygun, temiz ve optimize edilmiş Python kodu yazmak.',
    backstory="""Sen bir yazılım mimarısın. Analizlerini ve açıklamalarını 
    SADECE TÜRKÇE yaparsın. Kodun kalitesinden ödün vermezsin.""",
    llm=llama3,
    verbose=True
)

tester_agent = Agent(
    role='QA Test Mühendisi',
    goal='Yazılan kodu test etmek ve TÜRKÇE rapor sunmak.',
    backstory="""Sen titiz bir test mühendisisin. Raporlarını tamamen Türkçe hazırlarsın.""",
    llm=llama3,
    verbose=True
)

# 5. GÖREVLERİN TANIMLANMASI
coding_task = Task(
    description="{topic} gerçekleştiren bir Python scripti yaz. Açıklamalar Türkçe olmalı.",
    expected_output="Python kodu ve Türkçe kısa açıklamalar.",
    agent=coder_agent
)

testing_task = Task(
    description="Kodu test et ve Türkçe onay raporu hazırla.",
    expected_output="Türkçe test raporu ve onaylanmış kod.",
    agent=tester_agent
)

# 6. EKİBİ (CREW) OLUŞTURMA
dev_crew = Crew(
    agents=[coder_agent, tester_agent],
    tasks=[coding_task, testing_task],
    process=Process.sequential,
    memory=True,
    # Embedder ayarını doğrudan config ile de sağlamlaştırıyoruz
    embedder={
        "provider": "ollama",
        "config": {
            "model": "llama3",
            "base_url": "http://127.0.0.1:11434"
        }
    }
)

# 7. ÇALIŞTIRMA
print("\n### Ajanlar Çalışmaya Başlıyor...\n")
try:
    result = dev_crew.kickoff(inputs={'topic': 'FastAPI ve MongoDB kullanan bir CRUD API'})
    
    # Kayıt işlemi
    save_status = save_to_mongodb(result, "REST API Task")
    print(f"\n{save_status}")
    print("\n### FİNAL ÇIKTI:\n", result)
except Exception as e:
    print(f"Bir hata oluştu: {e}")