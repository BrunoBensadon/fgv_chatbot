from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun


search_tool = DuckDuckGoSearchRun()

#@tool
#def web_search_tool(query: str) -> str:
#    """Up-to-date web info via Tavily"""
#    try:
#        result = tavily.invoke({"query": query})
#
#        # Extract and format the results from Tavily response
#        if isinstance(result, dict) and 'results' in result:
#            formatted_results = []
#            for item in result['results']:
#                title = item.get('title', 'No title')
#                content = item.get('content', 'No content')
#                url = item.get('url', '')
#                formatted_results.append(f"Title: {title}\nContent: {content}\nURL: {url}")
#
#            return "\n\n".join(formatted_results) if formatted_results else "No results found"
#        else:
#            return str(result)
#    except Exception as e:
#        return f"WEB_ERROR::{e}"