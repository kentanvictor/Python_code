import urllib.request


class HtmlDownLoader(object):
    def download(self, url):
        if url is None:
            return None
        responce = urllib.request.urlopen(url)
        if responce.getcode() != 200:
            return None
        return responce.read()