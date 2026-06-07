import chromadb
from chromadb.config import Settings
import os

class RAGManager:
    def __init__(self, persist_directory="./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name="essay_memory")
        
    def add_critique(self, text_id: str, critique: str):
        """Adds a critique or specific learning point to memory."""
        self.collection.upsert(
            documents=[critique],
            metadatas=[{"type": "critique"}],
            ids=[text_id]
        )
        
    def add_winning_essay(self, text_id: str, text: str):
        """Adds a winning essay to memory as a reference."""
        self.collection.upsert(
            documents=[text],
            metadatas=[{"type": "winning_essay"}],
            ids=[text_id]
        )
        
    def get_relevant_memory(self, query: str, n_results: int = 3) -> str:
        """Retrieves relevant memories based on the query context."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            if not results['documents'] or not results['documents'][0]:
                return "No previous relevant critiques or guidelines found in memory."
                
            memory_context = "Past critiques and learnings to follow:\n"
            for doc in results['documents'][0]:
                memory_context += f"- {doc}\n"
            return memory_context
        except Exception as e:
            return f"Error retrieving memory: {e}"

# Example Usage
if __name__ == "__main__":
    rag = RAGManager()
    rag.add_critique("critique_1", "Selalu gunakan bahasa formal yang akademis dan hindari kata slang.")
    print(rag.get_relevant_memory("Bagaimana cara menulis essay yang baik?"))
