import os
from crewai import Agent, Task, Crew, Process, LLM

# 1. LLM AYARI (Ollama Llama3)
os.environ["OPENAI_API_KEY"] = "NA"

my_llm = LLM(
    model="ollama/llama3", 
    # Bağlantı hatası için localhost yerine 127.0.0.1 bazen daha stabildir
    base_url="http://127.0.0.1:11434" 
)

# 2. AJANLARIN TANIMLANMASI
researcher = Agent(
    role='Kıdemli Araştırma Analisti',
    goal='{topic} hakkındaki en son trendleri bul ve sadece Türkçe raporla',
    backstory="""Sen seçkin bir teknoloji araştırma merkezinde çalışan, 
    analizlerini SADECE TÜRKÇE dilinde yapan bir uzmansın. 
    Kesinlikle İngilizce terimler dışında Türkçe dışı cümle kurmazsın.""",
    llm=my_llm, 
    verbose=True,
    allow_delegation=False
)

writer = Agent(
    role='İçerik Stratejisti',
    goal='Araştırma raporuna dayanarak sadece Türkçe dilinde ilgi çekici bir blog yazısı yaz',
    backstory="""Sen teknik konuları Türk okuyucuların anlayabileceği şekilde 
    akıcı bir Türkçe ile hikayeleştiren ödüllü bir yazarsın. 
    Tüm içeriğin mükemmel bir Türkçe dil bilgisiyle yazılmış olmalıdır.""",
    llm=my_llm,
    verbose=True
)

# 3. GÖREVLERİN TANIMLANMASI
task1 = Task(
    description="{topic} hakkında 2026 yılı trendlerini analiz et. Yanıtın tamamen Türkçe olmalı.",
    expected_output="Türkçe dilinde hazırlanmış 5 maddelik detaylı bir trend raporu.",
    agent=researcher
)

task2 = Task(
    description="Trend raporunu kullanarak etkileyici bir blog yazısı oluştur. Yazı tamamen Türkçe olmalı.",
    expected_output="Markdown formatında, başlıkları ve içeriği tamamen Türkçe olan tam bir blog yazısı.",
    agent=writer
)

# 4. EKİBİN KURULMASI (CREW)
crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    process=Process.sequential
)

# 5. ÇALIŞTIRMA
# Not: Ollama uygulamasının açık olduğundan emin ol!
result = crew.kickoff(inputs={'topic': 'Yapay Zeka Ajanları'})

print("\n\n########################")
print("## TÜRKÇE SONUÇ:")
print("########################\n")
print(result)