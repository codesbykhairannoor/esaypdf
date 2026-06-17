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

        # STAGE 3: Iterative Chunking (Joki AI v7.0)
        logger.info(f"Stage 3: Iterative Drafting via Chunking")
        
        sections = [
            "Bagian 1: Pendahuluan dan Fakta Observasi/Tragedi Lapangan",
            "Bagian 2: Analisis Akar Masalah (Root-Cause)",
            "Bagian 3: Desain Solusi & Sistem Teknis",
            "Bagian 4: Hambatan Lapangan (Gesekan Implementasi) & Kesimpulan"
        ]
        
        chunk_draft_prompt = PromptTemplate.from_template(
            """{guidelines}
            
            Anda sedang menulis sebuah esai panjang berkaliber tinggi (Target total 2000 kata).
            Tema: {chosen_theme}
            Kerangka Keseluruhan: {outline}
            Referensi Akademik: {academic_references}
            
            TEKS YANG SUDAH DITULIS SEBELUMNYA (Lanjutkan dari sini):
            {previous_text}
            
            TUGAS ANDA SAAT INI:
            Tulislah SECARA EKSKLUSIF untuk **{current_section}**.
            Targetkan bagian ini minimal 400-500 kata yang padat, kritis, dan mendalam.
            
            ATURAN:
            1. Jangan mengulang pendahuluan jika ini bukan Bagian 1.
            2. Wajib mengutip referensi akademik di atas secara natural.
            3. DILARANG menggunakan kata-kata klise (paradoks, di satu sisi, status quo).
            4. Jangan membuat tabel di bagian ini, cukup fokus pada narasi paragraf.
            5. Lanjutkan alur cerita/argumen dengan mulus dari teks sebelumnya tanpa basa-basi pengantar yang mengulangi judul.
            """
        )
        
        chunks = []
        previous_text = "(Belum ada teks, ini bagian pertama)"
        
        for section in sections:
            logger.info(f"Drafting {section}...")
            chunk_content = (chunk_draft_prompt | self.llm).invoke({
                "guidelines": guidelines_text,
                "chosen_theme": chosen_theme,
                "outline": outline,
                "academic_references": academic_refs,
                "previous_text": previous_text[-3000:], # Keep only last 3000 chars as context to prevent token overflow
                "current_section": section
            }).content
            chunks.append(chunk_content)
            previous_text = "\n\n".join(chunks)
            
        full_draft = "\n\n".join(chunks)
        
        # STAGE 3.5: Dedicated Attachment Generation
        logger.info(f"Stage 3.5: Generating Attachments/Tables")
        attachment_prompt = PromptTemplate.from_template(
            """Berdasarkan draf esai berikut ini, buatkan LAMPIRAN berupa DUA BUAH TABEL MARKDOWN TERSTRUKTUR.
            Tabel 1: Perbandingan Konseptual
            Tabel 2: Roadmap Implementasi & Analisis Hambatan
            
            DRAF ESAI (Hanya sebagai konteks, tidak perlu ditulis ulang):
            {full_draft}
            
            ATURAN:
            - Hanya keluarkan format Markdown Tabel (dimulai dengan |).
            - Jangan tambahkan teks basa-basi lain di atas atau di bawah tabel.
            """
        )
        tables_content = (attachment_prompt | self.llm).invoke({"full_draft": full_draft[-4000:]}).content
        
        # Combine everything
        final_essay = full_draft + "\n\n## Lampiran\n\n" + tables_content
        
        # Post-Processing Anti-Template Filter
        logger.info("Applying Post-Processing Filter...")
        cliches = [
            (r"(?i)di satu (sisi|pihak)", "pada satu aspek"),
            (r"(?i)di (sisi|pihak) lain", "sementara itu"),
            (r"(?i)paradoks( kontemporer)?( yang tak terhindarkan)?", "tantangan nyata di lapangan"),
            (r"(?i)status quo", "kondisi saat ini"),
            (r"(?i)dilema", "persoalan mendasar"),
            (r"(?i)mutually exclusive(,)? (and )?collectively exhaustive", ""),
            (r"(?i)\(MECE\)", ""),
            (r"(?i)triple helix", "kolaborasi lintas pihak")
        ]
        for pattern, replacement in cliches:
            final_essay = re.sub(pattern, replacement, final_essay)
            
        # STAGE 4: Saving to DOCX
        logger.info("STAGE 4: Saving to DOCX")
        result_msg = create_docx(final_essay, output_path)
        logger.info(f"Finished: {result_msg}")
        return f"Essay generated successfully.\nDetails: {result_msg}"

    def learn_from_critique(self, critique_id: str, critique_text: str):
        self.memory.add_critique(critique_id, critique_text)
        return "Critique saved to memory. I will remember this for next time."
