# -*- coding: utf-8 -*-
import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from PExCrawler.items import PexcrawlerItem
import datetime
import urllib.parse
import sort_csv
from bs4 import BeautifulSoup

class PexcrawlerSpider(CrawlSpider):
    name = 'pexcrawler'
    allowed_domains = ['pinoyexchange.com']
    #start_urls = ['http://www.pinoyexchange.com/forums/forumdisplay.php?f=53&page=5']
    #start_urls = ['http://www.pinoyexchange.com/forums/forumdisplay.php?f=47']
    start_urls = ['http://www.pinoyexchange.com/forums/forumdisplay.php?f=10']
    # start_urls = ['http://www.pinoyexchange.com/forums/forumdisplay.php?f=53',
    #              'http://www.pinoyexchange.com/forums/forumdisplay.php?f=24',
    #              'http://www.pinoyexchange.com/forums/forumdisplay.php?f=91',
    #              'http://www.pinoyexchange.com/forums/forumdisplay.php?f=326',
    #              'http://www.pinoyexchange.com/forums/forumdisplay.php?f=51',
    #              'http://www.pinoyexchange.com/forums/forumdisplay.php?f=8',
    #              'http://www.pinoyexchange.com/forums/forumdisplay.php?f=307',
    #              'http://www.pinoyexchange.com/forums/forumdisplay.php?f=85',
    #              'http://www.pinoyexchange.com/forums/forumdisplay.php?f=94',
    #              'http://www.pinoyexchange.com/forums/forumdisplay.php?f=47']
    # ---------------------------------- CRAWLED --------------------------------------

    #start_urls = ['http://pinoyexchange.com/forumslist/'] # Add special forums (Universities)

    custom_settings = {
        'FEED_EXPORT_FIELDS': ['subforum', 'thread_title', 'post_subject', 'post_counter', 'post_content', 'post_time',
                               'username', 'user_posts_per_day', 'user_total_posts', 'quoted_post', 'quoted_username',
                               'img_urls', 'embed_urls', 'other_urls'],
    }

    # 'http://www.pinoyexchange.com/forums/forumdisplay.php?f=14',

    rules = (
        # Test for single thread
        # 280911 242700 538205 194397
        # 337799 - with @
        # 545556 - has quote, bold, urls, f&51
        # f&47, 794620 - has video, smilies, picture, quotes, post title, spoiler
        # Cars and motoring f&53, 434851 has multi-quote
        Rule(
            LinkExtractor(
                #allow=(['showthread\.php\?t=\d+', 'forumdisplay\.php\?f=\d+']),
                allow=('showthread\.php\?t=593662'),
                #allow=('showthread\.php\?t=438710'),
                #allow=(['showthread\.php\?t=893473', 'showthread\.php\?t=894225', 'showthread\.php\?t=411406']),
                #allow=('showthread\.php\?t=545556'),
                #allow=('showthread\.php\?t=794620'),
                restrict_xpaths=(
                    ['.//a[starts-with(@id, "thread_title")]',
                     './/a[@rel="next"]']
                ),
            ),
            callback='parse_item',
            follow=True
        ),
    )

    def has_smilies(self, img_link):
        if "images/smilies/" in img_link:
            return True
        return False

    def string_to_delta(self, string_delta):
        if string_delta == "yesterday":
            return (datetime.datetime.now() - datetime.timedelta(1)).strftime("%b %d, %Y")

        value, unit, _ = string_delta.split()
        unit_list = ['hour', 'day', 'week']
        if unit in unit_list:
            unit += 's'
        return (datetime.datetime.now() - datetime.timedelta(**{unit: float(value)})).strftime("%b %d, %Y")

    def parse_profile(self, response):
        item = response.meta['item']
        selector = response.css('div#view-stats')
        item['user_total_posts'] = selector.xpath('./div[2]/dl[1]/dd/text()').extract()
        item['user_posts_per_day'] = selector.xpath('./div[2]/dl[2]/dd/text()').extract()
        return item

    def parse_item(self, response):
        selector_list = response.css('li.postcontainer')

        for selector in selector_list:
            item = PexcrawlerItem()
            item['subforum'] = response.css('ul.floatcontainer').xpath('./li[4]/a/text()').extract()
            item['thread_title'] = [i.strip() for i in response.css('li.lastnavbit').xpath('./h1/text()').extract()]
            item['username'] = selector.xpath('normalize-space(./div[2]/div/div/div/a/strong//text())').extract()
            #item['post_time'] = [self.string_to_delta(s) if "ago" in s else s for s in selector.xpath('./div/span/span/text()').extract()]
            item['post_time'] = [self.string_to_delta(s) if "ago" in s or "yesterday" in s else s for s in
                                 selector.xpath('./div/span/span/text()').extract()]
            # Removes newlines
            #x = [i.strip() for i in selector.xpath('./div[2]/div[2]/div/div/div/blockquote/descendant::text()[not(ancestor::div/@class="bbcode_container")]').extract()]
            #item['post_content'] = ' '.join(list(filter(None, x)))
            html_post_content = selector.xpath('./div[2]/div[2]/div/div/div').extract()
            posts = []

            for post in html_post_content:
                soup = BeautifulSoup(post)
                [post.extract() for post in soup('div', {"class":"bbcode_container"})]
                content = soup.find('blockquote', {"class":["postcontent", "restore"]})
                posts.append(content.get_text(' ', strip=True))
                # print(content.get_text(' ', strip=True))
                # print('--------------------------------------------')
            item['post_content'] = posts
            item['post_counter'] = selector.xpath('./div/span[2]/a[2]/@name').extract()
            # //text for bolded text
            # y = [i.strip() for i in selector.xpath('./div[2]/div[2]/div/div/div/blockquote/div/div/div/div[3]/descendant::text()').extract()]
            # item['quoted_post'] = ' '.join(list(filter(None, y)))
            #item['quoted_post'] = selector.xpath('./div[2]/div[2]/div/div/div/blockquote/div/div/div/div[3]/descendant::text()').extract()
            #for s in selector.xpath('string(./div[2]/div[2]/div/div/div/blockquote/div/div/div/div[3])').extract():
            html_quoted_post = selector.xpath('./div[2]/div[2]/div/div/div/blockquote/div/div/div/div[3]').extract()
            q_posts = []
            divs = 0
            for s in html_quoted_post:
                soup = BeautifulSoup(s)
                [s.extract() for s in soup(['br', 'img'])]
                content = soup.find('div', {"class": "message"})
                divs += 1
                if divs > 1:
                    q_posts.append('<sep>')
                q_posts.append(content.get_text(' ', strip=True))
            item['quoted_post'] = q_posts
            item['quoted_username'] = selector.xpath('./div[2]/div[2]/div/div/div/blockquote/div/div/div/div[2]/strong//text()').extract()
            z = [i.strip() for i in selector.xpath('./div[2]/div[2]/div/h2/text()').extract()]
            item['post_subject'] = ' '.join(list(filter(None, z)))
            # filter smilies
            item['img_urls'] = [url for url in selector.xpath('./div[2]/div[2]/div/div/div/blockquote/img/@src').extract() if not self.has_smilies(url)]
            item['other_urls'] = selector.xpath('./div[2]/div[2]/div/div/div/blockquote/a/@href').extract()
            item['embed_urls'] = selector.xpath('./div[2]/div[2]/div/div/div/blockquote/iframe/@src').extract()

            member_url = urllib.parse.urljoin(response.url, selector.xpath('./div[2]/div/div/div/a/@href').extract_first())
            request = scrapy.Request(member_url, meta={'item': item}, callback=self.parse_profile, dont_filter=True)
            request.meta['item'] = item

            # How to get semantics? Bag of words tf/idf
            yield request

    def closed(self, reason):
        sort_csv.sort_data()
