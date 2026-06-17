import requests
import logging

logger = logging.getLogger(__name__)

def fetch_academic_papers(query: str, max_results: int = 3) -> str:
    """
    Fetches academic papers related to the query using the free CrossRef API.
    Returns a formatted string containing the titles, authors, years, and abstracts.
    """
    logger.info(f"Fetching academic papers from CrossRef for query: {query}")
    url = "https://api.crossref.org/works"
    params = {
        "query": query,
        "select": "title,abstract,author,issued",
        "rows": max_results
    }
    
    headers = {
        # Polite pool etiquette for CrossRef
        "User-Agent": "JokiEsaiBot/1.0 (mailto:admin@jokiesai.com)"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        items = data.get("message", {}).get("items", [])
        if not items:
            logger.warning("No academic papers found for the query.")
            return "Tidak ada referensi jurnal ditemukan."
            
        formatted_references = "REFERENSI JURNAL AKADEMIK (Gunakan data ini untuk mendukung argumen esai):\n\n"
        for i, item in enumerate(items, 1):
            title = item.get("title", ["Unknown Title"])[0]
            
            # Extract authors
            authors_list = item.get("author", [])
            authors = []
            for author in authors_list:
                family = author.get("family", "")
                given = author.get("given", "")
                name = f"{given} {family}".strip()
                if name:
                    authors.append(name)
            author_str = ", ".join(authors) if authors else "Unknown Author"
            
            # Extract year
            issued = item.get("issued", {})
            date_parts = issued.get("date-parts", [[]])
            year = "Unknown Year"
            if date_parts and len(date_parts[0]) > 0:
                year = str(date_parts[0][0])
                
            # Extract abstract (might contain XML/HTML tags)
            abstract = item.get("abstract", "Tidak ada abstrak.")
            # Clean up jats tags if any
            import re
            abstract_clean = re.sub(r'<[^>]+>', '', abstract)
            
            formatted_references += f"{i}. Judul: {title}\n"
            formatted_references += f"   Penulis: {author_str}\n"
            formatted_references += f"   Tahun: {year}\n"
            formatted_references += f"   Abstrak: {abstract_clean}\n\n"
            
        return formatted_references.strip()
        
    except Exception as e:
        logger.error(f"Failed to fetch academic papers: {e}")
        return f"Gagal mengambil referensi jurnal: {e}"

if __name__ == "__main__":
    # Test script
    logging.basicConfig(level=logging.INFO)
    print(fetch_academic_papers("pendidikan AI indonesia", 2))
