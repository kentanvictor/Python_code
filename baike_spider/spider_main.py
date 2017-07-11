# coding:utf8
from baike_spider import url_manager, html_downloader, html_parser, html_outputer


class SpiderMain(object):
    def __init__(self):
        self.urls = url_manager.UrlManager()
        self.downloader = html_downloader.HtmlDownLoader()
        self.parser = html_parser.HtmlParser()
        self.outputer = html_outputer.HtmlOutputer()

    def craw(self, root_url):
        count = 1
        self.urls.add_new_url(root_url)
        while self.urls.has_new_url():
            try:
                new_url = self.urls.get_new_url()
                print('craw %d : %s' % (count, new_url))
                html_cont = self.downloader.download(new_url)
                new_urls, new_data = self.parser.parse(new_url, html_cont)
                self.urls.add_new_urls(new_urls)
                self.outputer.collect_data(new_data)
                if count == 1000:
                    break
                count = count + 1
            except:
                print("craw fail")
                self.outputer.output_html()


if __name__ == "_main_":
    root_url = "http://baike.baidu.com/link?url=zsQ6Q3VSTmborakJcfOpLzkYdHXFagZR_Mez9Ol3_aP1KA_esSO0jzkRQS3BX3BT7Dg_L4H1P-U8g4zytzqQG_"
    obj_spider = SpiderMain()
    obj_spider.craw(root_url)
