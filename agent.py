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
        api_key = os.getenv("ALIBABA_API_KEY")
        self.llm = ChatOpenAI(
             model="qwen-plus",
             api_key=api_key,
             base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
             max_tokens=3000, 
             temperature=0.7
        )
        self.memory = RAGManager()
        
    def process_essay_request(self, user_prompt: str, pdf_path: str, output_path: str) -> str:
        """Deep research multi-stage pipeline with Judge Mode and Autonomy."""
        logger.info("STAGE 1: Reading Guidelines & Memory")
        guidelines = "No PDF provided."
        if pdf_path and os.path.exists(pdf_path):
            guidelines = extract_text_from_pdf(pdf_path)[:15000]
            
        past_learnings = self.memory.get_relevant_memory("Essay writing guidelines and rules")
        logger.info(f"Loaded past learnings: {past_learnings}")

        # STAGE 2: Outline, Theme Autonomy & Research Queries
        logger.info("STAGE 2: Generating Outline and Search Queries")
        strategy_prompt = PromptTemplate(
            input_variables=["user_prompt", "guidelines"],
            template="""You are an expert academic strategist and essay competition winner.
            USER REQUEST: {user_prompt}
            GUIDELINES: {guidelines}
            
            CRITICAL INSTRUCTION FOR THEME SELECTION:
            If the USER REQUEST does not specify a specific theme or subtheme, you MUST read the GUIDELINES, extract all available themes, and autonomously choose the ONE subtheme that is the most unique, innovative, and likely to win. State your chosen theme explicitly.
            
            Based on the request (or your autonomous theme choice) and guidelines, generate a detailed 3-part outline for the essay:
            1. Pendahuluan
            2. Isi (Gagasan Utama)
            3. Penutup
            
            Also, provide exactly 3 specific Google search queries needed to find real statistics or facts for this essay.
            Format your response strictly as:
            CHOSEN THEME: [Your chosen theme]
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
        logger.info("STAGE 4: Drafting the Initial Essay")
        draft_prompt = PromptTemplate(
            input_variables=["user_prompt", "guidelines", "past_learnings", "strategy", "research_data"],
            template="""You are an elite essay writer. Write the FIRST DRAFT of a highly academic essay.
            
            USER REQUEST: {user_prompt}
            GUIDELINES: {guidelines}
            PAST CRITIQUES TO AVOID: {past_learnings}
            
            APPROVED STRATEGY & OUTLINE:
            {strategy}
            
            RESEARCH DATA:
            {research_data}
            
            INSTRUCTIONS:
            - Write the complete essay draft.
            - Use proper Markdown styling: # for Main Title, ## for Sections, ** for bold.
            - Integrate the research data naturally.
            - DO NOT output anything except the essay content.
            """
        )
        
        draft_content = (draft_prompt | self.llm).invoke({
            "user_prompt": user_prompt,
            "guidelines": guidelines,
            "past_learnings": past_learnings,
            "strategy": strategy.split("QUERIES:")[0],
            "research_data": research_data
        }).content
        logger.info("Initial Draft Completed.")

        # STAGE 5: Judge Mode (Self-Evaluation)
        logger.info("STAGE 5: Judge Mode (Self-Evaluation)")
        judge_prompt = PromptTemplate(
            input_variables=["guidelines", "draft_content"],
            template="""You are a strict, world-class competition judge. 
            Review the following essay draft against the competition guidelines.
            
            GUIDELINES & RUBRIC:
            {guidelines}
            
            ESSAY DRAFT:
            {draft_content}
            
            INSTRUCTIONS:
            Identify flaws, weak arguments, formatting violations, or lack of data in the draft.
            Provide a brutally honest critique and specific instructions on how to improve it.
            """
        )
        critique = (judge_prompt | self.llm).invoke({
            "guidelines": guidelines,
            "draft_content": draft_content
        }).content
        logger.info(f"Judge Critique:\n{critique}")

        # STAGE 6: Final Revision
        logger.info("STAGE 6: Final Revision")
        revision_prompt = PromptTemplate(
            input_variables=["draft_content", "critique"],
            template="""You are the elite essay writer. You have received a harsh critique from the judge.
            
            INITIAL DRAFT:
            {draft_content}
            
            JUDGE's CRITIQUE:
            {critique}
            
            INSTRUCTIONS:
            Rewrite and polish the entire essay to perfectly address the judge's critique.
            Maintain excellent academic formatting (Markdown headers, bolding).
            DO NOT output anything except the final revised essay content.
            """
        )
        final_essay = (revision_prompt | self.llm).invoke({
            "draft_content": draft_content,
            "critique": critique
        }).content

        # STAGE 7: Saving to DOCX
        logger.info("STAGE 7: Saving to DOCX")
        result_msg = create_docx(final_essay, output_path)
        logger.info(f"Finished: {result_msg}")
        return f"Essay generated successfully.\nDetails: {result_msg}"

    def learn_from_critique(self, critique_id: str, critique_text: str):
        self.memory.add_critique(critique_id, critique_text)
        return "Critique saved to memory. I will remember this for next time."
