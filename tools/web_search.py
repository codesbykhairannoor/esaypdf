from duckduckgo_search import DDGS

def search_web(query: str, max_results: int = 3) -> str:
    """Performs a web search and returns the top results as a formatted string."""
    try:
        results = ""
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results += f"Title: {r['title']}\n"
                results += f"Snippet: {r['body']}\n"
                results += f"Link: {r['href']}\n\n"
        
        if not results:
            return "No results found."
        return results
    except Exception as e:
        return f"Error during web search: {e}"
