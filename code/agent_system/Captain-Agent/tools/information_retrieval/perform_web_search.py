
def perform_web_search(query, count=10, offset=0):
    """
    Perform a web search using Serper (Google) API.

    Args:
        query (str): The search query.
        count (int, optional): Number of search results to retrieve. Defaults to 10.
        offset (int, optional): Offset of the first search result. Defaults to 0.

    Returns:
        The name, URL and snippet of each search result.
    """
    import os
    import requests

    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        raise ValueError("SERPER_API_KEY not found in environment variables")

    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {
        "q": query,
        "num": count,
        "start": offset + 1,  # serper uses 1-based start
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("organic", [])
    for idx, item in enumerate(results):
        print(f"Search Result {idx+1}:")
        print(item.get("title"))
        print(item.get("link"))
        print(item.get("snippet"))
    return results
