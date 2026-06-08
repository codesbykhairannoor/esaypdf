from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup

def scrape_url_text(url: str, max_chars: int = 3000) -> str:
    """Visits a URL and extracts the main paragraph text."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Timeout to prevent hanging on slow sites
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract text from paragraphs
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text(strip=True) for p in paragraphs])
        
        if not text.strip():
            return "No readable text found on page."
            
        # Truncate to max_chars to save context window tokens
        return text[:max_chars] + "..." if len(text) > max_chars else text
        
    except Exception as e:
        return f"Failed to read page: {str(e)}"

def search_web(query: str, max_results: int = 3) -> str:
    """Performs a web search, visits the top URLs, and returns the full content."""
    try:
        results = ""
        with DDGS() as ddgs:
            search_items = list(ddgs.text(query, max_results=max_results))
            
            for r in search_items:
                results += f"=== Source Title: {r['title']} ===\n"
                results += f"URL: {r['href']}\n"
                results += f"Search Snippet: {r['body']}\n"
                
                # Perform Deep Scraping
                print(f"INFO: Deep scraping URL: {r['href']}")
                page_text = scrape_url_text(r['href'])
                results += f"Full Page Content:\n{page_text}\n\n"
        
        if not results:
            return "No results found."
        return results
    except Exception as e:
        return f"Error during web search: {e}"
