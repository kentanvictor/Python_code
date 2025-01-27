from bs4 import BeautifulSoup
import re

html_doc = """
<html><head><title>The Dormouse's story</title></head>
<body>
<p class="title"><b>The Dormouse's story</b></p>
<p class="story">Once upon a time there were three little sisters; and their names were
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>
<p class="story">...</p>
"""
soup = BeautifulSoup(html_doc, 'html.parser', from_encoding='utf-8')
print('获取所有链接')
links = soup.find_all('a')
for link in links:
    print(link.name, link['href'], link.get_text())
# 这里只想获取Lacie的链接，所以href后面的链接就可以直接复制过来
print('获取Lacie链接')
linknode = soup.find_all('a', href='http://example.com/lacie')
for link in linknode:
    print(link.name, link['href'], link.get_text())
# 正则匹配就相当于模糊匹配
print('正则匹配')
linknode = soup.find_all('a', href=re.compile(r'ill'))
for link in linknode:
    print(link.name, link['href'], link.get_text())
print('获取P')
pnode = soup.find_all('p', class_='title')
for link in pnode:
    print(link.name, link.get_text())
