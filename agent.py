import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from memory.rag_manager import RAGManager
from tools.web_search import search_web
from tools.pdf_reader import extract_text_from_pdf
from tools.docx_writer import create_docx

load_dotenv()

class EssayAgent:
    def __init__(self):
        # Initialize Deepseek LLM
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key or api_key == "sk-c863167f0690432aa2587ff12ebbcbd3":
             print("INFO: Menggunakan API Key DeepSeek dari file .env.")
             
        self.llm = ChatOpenAI(
             model="deepseek-chat",
             api_key=api_key,
             base_url="https://api.deepseek.com/v1",
             max_tokens=2500,  # Membatasi token untuk menghemat biaya
             temperature=0.7
        )
        self.memory = RAGManager()
        
    def process_essay_request(self, user_prompt: str, pdf_path: str, output_path: str) -> str:
        """Main pipeline to process an essay request."""
        # 1. Read PDF guidelines
        guidelines = "No PDF provided."
        if pdf_path and os.path.exists(pdf_path):
            guidelines = extract_text_from_pdf(pdf_path)
            
        # 2. Get past critiques (Memory)
        past_learnings = self.memory.get_relevant_memory("Essay writing guidelines and rules")
        
        # 3. Setup Prompt
        prompt = PromptTemplate(
            input_variables=["user_prompt", "guidelines", "past_learnings"],
            template="""You are an expert essay writer for competitions.
            
            USER REQUEST:
            {user_prompt}
            
            PDF GUIDELINES & RUBRIC (Read carefully and follow all constraints):
            {guidelines}
            
            PAST CRITIQUES TO REMEMBER (Do not repeat these mistakes):
            {past_learnings}
            
            Instructions:
            Write a complete, high-quality essay based on the guidelines.
            Use Markdown headers (e.g., # Title, ## Section) for structure.
            If you need facts, integrate them logically.
            """
        )
        
        # 4. Generate Essay
        chain = prompt | self.llm
        essay_content = chain.invoke({
            "user_prompt": user_prompt,
            "guidelines": guidelines[:15000], # limit context if too long
            "past_learnings": past_learnings
        }).content
        
        # 5. Save to DOCX
        result_msg = create_docx(essay_content, output_path)
        return f"Essay generated successfully.\nDetails: {result_msg}"

    def learn_from_critique(self, critique_id: str, critique_text: str):
        """Saves a critique to memory."""
        self.memory.add_critique(critique_id, critique_text)
        return "Critique saved to memory. I will remember this for next time."
