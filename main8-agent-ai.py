import os
from crewai import Agent, Task, Crew, Process, LLM
from pymongo import MongoClient

# 1. ORTAM VE LLM YAPILANDIRMASI
os.environ["OPENAI_API_KEY"] = "NA"
os.environ["EMBEDDINGS_OLLAMA_MODEL_NAME"] = "llama3" 

llama3 = LLM(
    model="ollama/llama3",
    base_url="http://127.0.0.1:11434"
)

# 2. MONGODB BAĞLANTISI
try:
    username = "root11"
    password = "371root1124"
    # authSource=admin eklemeyi unutma
    MONGO_URI = f"mongodb://{username}:{password}@localhost:27017/rag_pipeline?authSource=admin"
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    db = mongo_client["code_factory"]
    collection = db["saved_codes"]
    mongo_client.server_info()
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

# 4. AJANLARIN TANIMLANMASI (BASİTLEŞTİRİLDİ)
coder_agent = Agent(
    role='Python Yazılımcısı',
    goal='{topic} için en fazla 3 satırlık çok basit bir Python kodu yazmak.',
    backstory="""Sen minimalist bir yazılımcısın. Sadece işi yapan en kısa 
    kodu yazarsın. Asla uzun açıklamalar yapmazsın, SADECE TÜRKÇE konuşursun.""",
    llm=llama3,
    verbose=True
)

tester_agent = Agent(
    role='Kod Denetçisi',
    goal='Yazılan kodu kontrol edip "UYGUN" veya "HATALI" demek.',
    backstory="""Çok az konuşan bir denetçisin. Sadece kısa bir Türkçe onay verirsin.""",
    llm=llama3,
    verbose=True
)

# 5. GÖREVLERİN TANIMLANMASI (KISITLAMALAR EKLENDİ)
coding_task = Task(
    description="""{topic} konusunu yapan bir kod yaz. 
    KURAL: Kod 3 satırı geçmesin. Uzun açıklama yapma.""",
    expected_output="En fazla 3 satırlık Python kodu ve tek cümlelik Türkçe açıklama.",
    agent=coder_agent
)

testing_task = Task(
    description="Kodun çalışabilirliğini tek kelimeyle kontrol et.",
    expected_output="Kısa bir Türkçe onay mesajı.",
    agent=tester_agent
)

# 6. EKİBİ (CREW) OLUŞTURMA
dev_crew = Crew(
    agents=[coder_agent, tester_agent],
    tasks=[coding_task, testing_task],
    process=Process.sequential,
    memory=False, # BİLGİSAYARI RAHATLATMAK İÇİN: Hafızayı kapattık (RAM tasarrufu)
    embedder={
        "provider": "ollama",
        "config": {"model": "llama3"}
    }
)

# 7. ÇALIŞTIRMA VE AKILLI KAYIT
print("\n### Hafif Modda Ajanlar Başlıyor...\n")
try:
    # Ekibi çalıştır
    # crew_output = dev_crew.kickoff(inputs={'topic': 'Bir listenin ortalamasını alan fonksiyon'})
    crew_output = dev_crew.kickoff(inputs={'topic': '100 e kadar olan sayılar arasındaki en büyük asal sayıyı bulan fonksiyon'})
    # Task çıktılarına erişim: 
    # İlk görevde (coding_task) üretilen asıl kodu alıyoruz
    generated_code = coding_task.output.raw
    # İkinci görevdeki (testing_task) onay raporunu alıyoruz
    test_report = testing_task.output.raw

    # MongoDB'ye sadece "UYGUN" değil, asıl kodu kaydediyoruz
    final_save_content = f"YAZILAN KOD:\n{generated_code}\n\nTEST RAPORU: {test_report}"
    
    save_status = save_to_mongodb(final_save_content, "Basit Islem Task")
    
    print(f"\n{save_status}")
    print("\n### FİNAL ÇIKTI (DB'YE KAYDEDİLEN):\n", final_save_content)

except Exception as e:
    print(f"Bir hata oluştu: {e}")