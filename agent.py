import os
import logging
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from memory.rag_manager import RAGManager
from tools.pdf_reader import extract_text_from_pdf
from tools.docx_writer import create_docx
from tools.academic_researcher import fetch_academic_papers


load_dotenv()
logger = logging.getLogger(__name__)

class EssayAgent:
    def __init__(self):
        api_key = os.getenv("ALIBABA_API_KEY")
        models_to_try = [
            "qwen3-max",
            "qwen3.6-flash",
            "qwen3.6-max-preview",
            "qwen3.6-plus",
            "qwen3.5-397b-a17b",
            "deepseek-v3.2",
            "qwen3.5-122b-a10b"
        ]
        
        llms = []
        for model_name in models_to_try:
            llms.append(ChatOpenAI(
                model=model_name,
                api_key=api_key,
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                temperature=0.7,
                max_retries=0,  # Fail over immediately if a model fails or times out
                timeout=120,    # Increase timeout to 120s to allow search + long generation
                max_tokens=4000, # Allow large token output for long detailed essays
                extra_body={"enable_search": True}
            ))
            
        self.llm = llms[0].with_fallbacks(fallbacks=llms[1:])
        self.memory = RAGManager()
        
    def process_essay_request(self, user_prompt: str, pdf_path: str, output_path: str) -> str:
        """Deep research multi-stage pipeline with Judge Mode, Autonomy, and Qwen Optimizations."""
        logger.info("STAGE 1: Reading Guidelines & Memory")
        guidelines_text = "No PDF provided."
        if pdf_path and os.path.exists(pdf_path):
            guidelines_text = extract_text_from_pdf(pdf_path)[:15000]
            
        past_learnings = self.memory.get_relevant_memory("Essay writing guidelines and rules")
        logger.info(f"Loaded past learnings: {past_learnings}")

        # STAGE 2: Strategy & Outline Planning
        logger.info(f"Stage 2: Formulating strategy and outline.")
        strategy_prompt = PromptTemplate.from_template(
            """{guidelines}
            
            Berdasarkan Panduan Lomba di atas, bertindaklah sebagai Profesor Akademik jenius dari Ivy League yang kritis, berani, dan tajam.
            Tugas pertama Anda adalah menganalisis panduan dan memilih satu tema spesifik yang paling unik, berani, dan berpotensi menang.
            Gunakan logika berpikir terstruktur untuk membuat kerangka essay (Outline) yang komprehensif tanpa ada argumen yang tumpang tindih, namun **gaya penulisan harus mengalir secara organik, tidak kaku atau seragam seperti template AI biasa**.
            
            Aturan Mutlak Penulisan (Humanisasi Bahasa & Grounding Data):
            1. **Hancurkan Struktur Template AI (Cookie-Cutter)**: DILARANG menggunakan kerangka 3 dimensi (Sosial, Ekonomi, Teknologi) atau Roadmap 3 Fase (Pendek, Menengah, Panjang). Gunakan struktur naratif investigatif: (1) Kasus Kegagalan/Tragedi Spesifik di Lapangan, (2) Analisis Akar Masalah (Root-Cause), (3) Desain Sistem Teknis Detail, (4) Gesekan Implementasi Lokal.
            2. **Hapus Label Framework Eksplisit**: JANGAN menyebutkan atau menuliskan nama framework secara eksplisit di dalam esai (Dilarang menuliskan kata-kata seperti: *MECE*, *mutually exclusive*, *Triple Helix*, dll.). Biarkan pembagian argumen mengalir logis secara implisit/alami.
            3. **Gaya Pembuka Anti-Robot**: DILARANG KERAS menggunakan frasa klise khas AI (DILARANG pakai: *paradoks*, *dilema*, *status quo*, *di satu sisi... di sisi/pihak lain*). MULAI esai dengan observasi lapangan keras, anomali kasus lokal, atau kutipan realistis dari aktor di lapangan.
            3. **Kevalidan Fakta & Presisi Lokal**: Wajib memvalidasi fakta secara akurat. JANGAN PERNAH mengaitkan isu hoaks/siber dengan BNPB. Gunakan Kementerian Kominfo, MAFINDO, atau TurnBackHoax. Pastikan detail penyelenggara (Pusdima Unmul, Universitas Mulawarman, Samarinda) ditulis dengan presisi 100%. JANGAN mengarang data statistik atau persentase lembaga tanpa dasar.
            4. **Suara Penulis yang Tajam (Bold Voice)**: Ambil posisi/opini kritis yang berani atas isu hangat ini. Gaya tulisan harus memiliki karakter, "jiwa", sentuhan emosional (pathos) yang kuat, dan persuasif, bukan netral seperti laporan korporat.
            5. **Solusi Realistis dengan Hambatan (Feasibility)**: Hindari solusi utopis yang mengasumsikan segalanya berjalan mulus. Outline harus menyertakan bab/analisis khusus mengenai hambatan nyata di lapangan (misal: resistensi sosial, keterbatasan anggaran, keengganan birokrasi, risiko etis) serta *mitigasi konkret* untuk menghadapinya.
            6. **Ketajaman Teknis Skala Mahasiswa**: Hindari klaim teknologi bombastis yang tidak masuk akal untuk kapasitas mahasiswa (JANGAN menyebutkan penggunaan *Meta API* atau *TikTok API* untuk riset kecil). Ganti dengan metode yang riil seperti observasi digital manual, kuesioner acak, Small Language Model (SLM) lokal, atau RAG.
            7. **Safety Guardrails**: Sertakan analisis Safety Guardrails (batasan keamanan AI dari bias respon) serta mitigasi risiko hukum berupa pengalihan otomatis ke ahli/konselor manusia jika mendeteksi kondisi darurat.
            8. **Teologi Islam & AI**: Bahas batas teologis AI sebagai Wasilah (sarana), yang memicu kebutuhan mengembalikan hubungan dakwah sejati (Sanad, Ta'dzim, Qalb) ke Ulama/Guru manusia.
            9. **Pendukung Visual (Tabel)**: Buat bagian khusus di dalam outline untuk menyertakan minimal dua tabel markdown (Tabel Perbandingan Konseptual & Tabel Roadmap/SWOT dengan Friksi).
            
            Instruksi Pengguna tambahan: {user_prompt}
            
            Berikan output dalam format persis seperti ini:
            CHOSEN THEME: [Tema spesifik yang lu pilih]
            OUTLINE:
            (Tulis kerangka bab per bab dengan gaya MECE)
            """
        )
        strategy = (strategy_prompt | self.llm).invoke({"user_prompt": user_prompt, "guidelines": guidelines_text}).content
        logger.info(f"Strategy Output:\n{strategy}")

        # Extract chosen theme
        theme_match = re.search(r"CHOSEN THEME:(.*?)\n", strategy, re.IGNORECASE)
        chosen_theme = theme_match.group(1).strip() if theme_match else "Tema Umum"
        
        # Extract outline
        outline_match = re.search(r"OUTLINE:(.*)", strategy, re.DOTALL | re.IGNORECASE)
        outline = outline_match.group(1).strip() if outline_match else strategy

        # STAGE 2.5: Academic Research
        logger.info(f"Stage 2.5: Fetching real academic papers for theme: {chosen_theme}")
        academic_refs = fetch_academic_papers(chosen_theme, max_results=3)
        logger.info(f"Academic References Found:\n{academic_refs}")

        # STAGE 3 & 4: Drafting via Qwen Native Search
        logger.info(f"Stage 3 & 4: Drafting Content natively via Qwen Search")
        draft_prompt = PromptTemplate.from_template(
            """{guidelines}
            
            Berdasarkan Panduan Lomba di atas, buatlah Draft Lengkap Essay akademis tingkat tinggi dengan bobot materi yang sangat mendalam, tajam, dan panjang (Target: minimal 2000-2500 kata).
            
            Tema Terpilih: {chosen_theme}
            Kerangka Essay: 
            {outline}
            
            {academic_references}
            
            Aturan Penulisan Draf:
            1. **Gunakan Struktur Investigatif**: Wajib menggunakan alur: (1) Kasus Kegagalan/Tragedi Spesifik di Lapangan -> (2) Analisis Akar Masalah -> (3) Desain Sistem Teknis -> (4) Gesekan Implementasi Lokal.
            2. **Hapus Label Framework Eksplisit**: JANGAN memunculkan nama framework (*MECE*, *Triple Helix*, dll.).
            3. **Gaya Pembuka Anti-Robot**: DILARANG KERAS menggunakan kata klise AI (*paradoks*, *dilema*, *status quo*, *di satu sisi/pihak...*). MULAI setiap bab dengan fakta lapangan yang keras atau kutipan wawancara.
            4. **Skala Teknis Realistis**: Hindari klaim bombastis. Gunakan metode riset yang masuk akal dan riil seperti observasi digital manual atau kuesioner acak.
            5. **Operasionalisasi Ide Mendalam**: Pertebal pembahasan dengan menjelaskan secara konkret: ilustrasi nyata, skema alur sistem, parameter etika teknis, atau rancangan alur sistem. Jangan gunakan kalimat bertele-tele.
            6. **Kevalidan Data**: Gunakan Kominfo, MAFINDO, atau TurnBackHoax. JANGAN mengarang angka persentase fiktif.
            7. **Gaya Bahasa yang Bold**: Tulis dengan "jiwa" dan karakter penulis yang kuat, tajam, persuasif. Hindari bahasa laporan asisten AI.
            8. **Solusi Realistis & Analisis Hambatan**: Uraikan secara jujur tantangan nyata di lapangan beserta rencana mitigasi taktis yang konkret.
            9. **Spesifikasi SLM+RAG & Guardrails**: Usulkan Small Language Model (SLM) + RAG serta detail fungsi Safety Guardrails.
            10. **Aspek Teologi Wasilah**: Bahas batas teologis AI sebagai Wasilah.
            11. **Tabel Pendukung**: Wajib membuat minimal dua tabel markdown terstruktur.
            12. **Sitasi Jurnal Asli**: Anda WAJIB mengutip data, argumen, atau temuan dari "REFERENSI JURNAL AKADEMIK" yang disediakan di atas ke dalam esai, dan cantumkan sebagai daftar pustaka.
            
            Gunakan gaya bahasa akademik formal namun berkarakter kuat, dan gunakan aturan sitasi Harvard jika merujuk data. Tulis esai secara utuh mulai dari Judul sampai Kesimpulan.
            """
        )
        draft_content = (draft_prompt | self.llm).invoke({
            "guidelines": guidelines_text,
            "chosen_theme": chosen_theme,
            "outline": outline,
            "academic_references": academic_refs
        }).content
        logger.info("Initial Draft Completed.")

        # STAGE 5: Judge Mode (Self-Evaluation)
        logger.info(f"Stage 5: Judge Mode")
        judge_prompt = PromptTemplate.from_template(
            """{guidelines}
            
            Anda adalah Dewan Juri killer untuk lomba esai di atas. Evaluasi draft esai berikut ini dengan standar sangat tinggi.
            
            DRAFT ESSAY:
            {draft}
            
            Berikan kritik pedas mengenai koherensi MECE, tata bahasa, kekuatan argumen, dan orisinalitas ide. Apa yang kurang?
            """
        )
        critique = (judge_prompt | self.llm).invoke({"guidelines": guidelines_text, "draft": draft_content}).content
        logger.info(f"Judge Critique:\n{critique}")

        # STAGE 6: Final Revision
        logger.info(f"Stage 6: Revision")
        revision_prompt = PromptTemplate.from_template(
            """{guidelines}
            
            Anda adalah Penulis Ahli dengan gaya bahasa yang tajam, berani, dan berkarakter kuat. Revisi draft essay berikut berdasarkan kritik dari Juri.
            Pastikan output akhir ini mematuhi kaidah bahasa baku, terstruktur, dan mengandung argumen yang sangat meyakinkan serta tajam (Target panjang: 2000-2500 kata).
            
            Aturan Mutlak Revisi (Anti-AI Flaws):
            1. **Hancurkan Struktur Template AI**: Jika draft masih memakai kerangka 3 dimensi (Sosial/Ekonomi/Teknologi) atau Roadmap 3 Fase, ROMBAK TOTAL menjadi narasi berbasis Kasus Lapangan -> Desain Teknis -> Gesekan Implementasi.
            2. **Hapus Jargon Eksplisit**: JANGAN PERNAH menuliskan kata-kata seperti *MECE*, *Triple Helix*, dll.
            3. **Hancurkan Gaya Bahasa AI (Anti-Template)**: Bantai dan ganti semua kata klise (*paradoks*, *dilema*, *status quo*, *di satu sisi/pihak... di sisi/pihak lain*, *implikasi struktural dari...*). Ganti dengan narasi observasi riil lapangan.
            3. **Skala Teknis Realistis**: Pastikan metode penelitian yang disebut masuk akal (JANGAN sebut penggunaan *Meta API* atau *TikTok API*; ganti dengan observasi digital manual atau kuesioner acak).
            4. **Operasionalisasi Konkret (Anti-Fluff)**: Jangan memperpanjang teks dengan kalimat bertele-tele atau pengulangan ide. Perluas bab dengan merinci skema alur sistem, ilustrasi nyata, detail etika parameter teknis, dan mitigasi risiko.
            5. **Fakta & Lembaga Siber**: Koreksi semua halusinasi data (JANGAN hubungkan BNPB dengan isu hoaks siber keagamaan; ganti dengan MAFINDO, Kominfo, atau TurnBackHoax). Pastikan nama universitas dan kota penyelenggara (Pusdima Unmul, Universitas Mulawarman, Samarinda) tertulis 100% presisi.
            6. **Suara Penulis yang Bold**: Pertahankan nada tulisan yang berani mengambil posisi kritis atas isu keagamaan dan AI, bukan hambar atau netral.
            7. **Solusi Realistis & Mitigasi Risiko**: Pastikan esai tidak terdengar utopis. Harus ada penjelasan hambatan lapangan (birokrasi, dana, resistensi kultural) dan mitigasi konkretnya.
            8. **Spesifikasi SLM+RAG & Guardrails**: Tegaskan keunggulan Small Language Model (SLM) + RAG, dan jelaskan detail fungsi Safety Guardrails demi mitigasi risiko hukum (sistem pemutus otomatis yang mengalihkan user ke ahli/konselor manusia).
            9. **Teologi Wasilah**: Tegaskan peran AI sebagai Wasilah yang tetap membutuhkan bimbingan spiritual manusia (Sanad, Ta'dzim, Qalb).
            10. **Tabel Wajib**: Wajib sertakan minimal dua tabel markdown. JANGAN hapus format tabel markdown (`|---|`), pastikan tabel tetap utuh karena akan dikonversi menjadi tabel Word asli.
            
            Format teks hasil menggunakan format Markdown standar (Gunakan ** untuk tebal, * untuk miring).
            
            DRAFT LAMA:
            {draft}
            
            KRITIK JURI:
            {critique}
            
            MEMORI KRITIK MASA LALU (Terapkan jika relevan):
            {past_memories}
            
            Hasilkan Essay Final (tanpa catatan tambahan, langsung dari Judul sampai selesai).
            """
        )
        final_essay = (revision_prompt | self.llm).invoke({
            "guidelines": guidelines_text,
            "draft": draft_content,
            "critique": critique,
            "past_memories": past_learnings
        }).content

        # STAGE 7: Saving to DOCX
        logger.info("STAGE 7: Saving to DOCX")
        result_msg = create_docx(final_essay, output_path)
        logger.info(f"Finished: {result_msg}")
        return f"Essay generated successfully.\nDetails: {result_msg}"

    def learn_from_critique(self, critique_id: str, critique_text: str):
        self.memory.add_critique(critique_id, critique_text)
        return "Critique saved to memory. I will remember this for next time."
