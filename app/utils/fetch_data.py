from bs4 import BeautifulSoup
import requests
import re
from app.config.constant import DATASET_PATH
from collections import deque

class FetchData:
    def get_data_page(url: str, selector: str = "h1, h2, h3, h4, h5, p"):
        response = requests.get(url)
        response.raise_for_status()
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')

        results = ""
        for element in soup.select(selector):
            text = element.get_text(" ", strip=True)
            text = " ".join(text.split())
            results += " " + text
        return results
    
    def get_link(url: str, prefix: str, pattern: re.compile):
        response = requests.get(url)
        response.raise_for_status()
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        links = [a_tag['href'] for a_tag in soup.find_all('a', href=True) if pattern.search(a_tag['href'])]
        for i, link in enumerate(links):
            if link.startswith(('http:/','https:/')):
                continue
            else:
                links[i] = prefix + links[i]
        return links

    def get_data(
            url: str, 
            prefix: str = 'https://thanhnien.vn', 
            pattern: re.Pattern = re.compile(r'/.*-(\d{18})\.htm$|/(\d{18})\.htm$'), 
            depth: int = 1
    ):
        
        stack = deque()
        stack.append((url, 0))
        visited = {url}
        text = ""
        while stack :
            url, dep = stack.pop()
            if dep != 0:
                text += FetchData.get_data_page(url) + '\n'
            if dep == depth:
                continue
            urls_next = FetchData.get_link(url, prefix, pattern)

            for url_next in urls_next:
                if url_next not in visited:
                    stack.append((url_next, dep + 1))
                    visited.add(url_next)
        return text
    def write_file(data: str, filename):
        open(file=filename, mode='a', encoding='utf-8').write(data)
            
    # def get_data(root: str, prefix: str, depth: int = 1):