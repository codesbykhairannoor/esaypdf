import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from memory.rag_manager import RAGManager
from tools.web_search import search_web
from tools.pdf_reader import extract_text_from_pdf
from tools.docx_writer import create_docx

load_dotenv()
logger = logging.getLogger(__name__)

class EssayAgent:
    def __init__(self):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        self.llm = ChatOpenAI(
             model="deepseek-chat",
             api_key=api_key,
             base_url="https://api.deepseek.com/v1",
             max_tokens=3000, 
             temperature=0.7
        )
        self.memory = RAGManager()
        
    def process_essay_request(self, user_prompt: str, pdf_path: str, output_path: str) -> str:
        """Deep research multi-stage pipeline."""
        logger.info("STAGE 1: Reading Guidelines & Memory")
        guidelines = "No PDF provided."
        if pdf_path and os.path.exists(pdf_path):
            guidelines = extract_text_from_pdf(pdf_path)[:15000]
            
        past_learnings = self.memory.get_relevant_memory("Essay writing guidelines and rules")
        logger.info(f"Loaded past learnings: {past_learnings}")

        # STAGE 2: Outline & Research Queries
        logger.info("STAGE 2: Generating Outline and Search Queries")
        strategy_prompt = PromptTemplate(
            input_variables=["user_prompt", "guidelines"],
            template="""You are an expert academic strategist.
            USER REQUEST: {user_prompt}
            GUIDELINES: {guidelines}
            
            Based on the request and guidelines, generate a detailed 3-part outline for the essay:
            1. Pendahuluan
            2. Isi (Gagasan Utama)
            3. Penutup
            
            Also, provide exactly 3 specific Google search queries needed to find real statistics or facts for this essay.
            Format your response strictly as:
            OUTLINE:
            [Your outline here]
            QUERIES:
            1. [Query 1]
            2. [Query 2]
            3. [Query 3]
            """
        )
        strategy = (strategy_prompt | self.llm).invoke({"user_prompt": user_prompt, "guidelines": guidelines}).content
        logger.info(f"Strategy Output:\n{strategy}")

        # Extract Queries
        queries = []
        if "QUERIES:" in strategy:
            q_text = strategy.split("QUERIES:")[1].strip().split('\n')
            for line in q_text:
                if len(line) > 3 and line[0].isdigit():
                    queries.append(line.split('.', 1)[1].strip())
        
        # STAGE 3: Web Research
        logger.info(f"STAGE 3: Executing Web Research for queries: {queries}")
        research_data = ""
        for q in queries[:3]:
            logger.info(f"Searching: {q}")
            res = search_web(q)
            research_data += f"\nSearch '{q}':\n{res}\n"
            
        # STAGE 4: Drafting
        logger.info("STAGE 4: Drafting the Full Essay")
        draft_prompt = PromptTemplate(
            input_variables=["user_prompt", "guidelines", "past_learnings", "strategy", "research_data"],
            template="""You are an elite essay writer. Write a highly academic, flawless essay.
            
            USER REQUEST: {user_prompt}
            GUIDELINES: {guidelines}
            PAST CRITIQUES TO AVOID: {past_learnings}
            
            APPROVED OUTLINE:
            {strategy}
            
            RESEARCH DATA (Use these facts to strengthen your arguments):
            {research_data}
            
            INSTRUCTIONS:
            - Write the complete essay.
            - Use proper Markdown styling: # for Main Title, ## for Sections, ** for bold, and bullet points.
            - Integrate the research data naturally with academic tone.
            - Ensure smooth transitions between paragraphs.
            - DO NOT output anything except the essay content.
            """
        )
        
        essay_content = (draft_prompt | self.llm).invoke({
            "user_prompt": user_prompt,
            "guidelines": guidelines,
            "past_learnings": past_learnings,
            "strategy": strategy.split("QUERIES:")[0],
            "research_data": research_data
        }).content
        
        logger.info("STAGE 5: Saving to DOCX")
        result_msg = create_docx(essay_content, output_path)
        logger.info(f"Finished: {result_msg}")
        return f"Essay generated successfully.\nDetails: {result_msg}"

    def learn_from_critique(self, critique_id: str, critique_text: str):
        self.memory.add_critique(critique_id, critique_text)
        return "Critique saved to memory. I will remember this for next time."
