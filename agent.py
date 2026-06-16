import os
import logging
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from memory.rag_manager import RAGManager
from tools.pdf_reader import extract_text_from_pdf
from tools.docx_writer import create_docx

load_dotenv()
logger = logging.getLogger(__name__)

class EssayAgent:
    def __init__(self):
        api_key = os.getenv("ALIBABA_API_KEY")
        models_to_try = [
            "deepseek-v3.2",
            "qwen3.6-max-preview",
            "qwen3.6-plus",
            "qwen3-max",
            "qwen3.5-397b-a17b",
            "qwen3.6-flash",
            "qwen3.5-122b-a10b"
        ]
        
        llms = []
        for model_name in models_to_try:
            llms.append(ChatOpenAI(
                model=model_name,
                api_key=api_key,
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                temperature=0.7,
                max_retries=1,
                timeout=60,
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
            
            Berdasarkan Panduan Lomba di atas, bertindaklah sebagai Profesor Akademik jenius dari Ivy League.
            Tugas pertama Anda adalah menganalisis panduan dan memilih satu tema spesifik yang paling unik dan berpotensi menang.
            Gunakan framework MECE (Mutually Exclusive, Collectively Exhaustive) untuk membuat kerangka essay (Outline) yang komprehensif tanpa ada argumen yang tumpang tindih.
            
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

        # STAGE 3 & 4: Drafting via Qwen Native Search
        logger.info(f"Stage 3 & 4: Drafting Content natively via Qwen Search")
        draft_prompt = PromptTemplate.from_template(
            """{guidelines}
            
            Berdasarkan Panduan Lomba di atas, buatlah Draft Lengkap Essay akademis tingkat tinggi.
            
            Tema Terpilih: {chosen_theme}
            Kerangka Essay (MECE): 
            {outline}
            
            Karena kamu memiliki akses internet terintegrasi, carilah fakta, statistik, dan penelitian terbaru secara otomatis untuk mendukung argumen ini.
            Gunakan gaya bahasa akademik, formal, objektif, dan logis. Hindari kata klise. Gunakan transisi kalimat yang canggih.
            Gunakan aturan sitasi Harvard jika mengambil data.
            Tulis draft essay secara utuh mulai dari Judul, Pendahuluan, Isi, hingga Kesimpulan.
            """
        )
        draft_content = (draft_prompt | self.llm).invoke({
            "guidelines": guidelines_text,
            "chosen_theme": chosen_theme,
            "outline": outline
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
            
            Anda adalah Penulis Ahli. Revisi draft essay berikut berdasarkan kritik dari Juri.
            Pastikan output akhir ini sudah mematuhi kaidah bahasa baku, terstruktur dengan metode MECE, dan mengandung argumen yang sangat meyakinkan.
            Format teks hasil menggunakan format Markdown standar (Gunakan ** untuk tebal, * untuk miring).
            
            DRAFT LAMA:
            {draft}
            
            KRITIK JURI:
            {critique}
            
            MEMORI KRITIK MASA LALU (Terapkan jika relevan):
            {past_memories}
            
            Hasilkan Essay Final (tanpa catatan tambahan).
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
