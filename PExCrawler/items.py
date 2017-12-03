# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

from scrapy import Item, Field


class PexcrawlerItem(Item):
    # define the fields for your item here like:
    subforum = Field()
    thread_title = Field()
    username = Field()
    post_time = Field()
    post_content = Field()
    post_counter = Field()
    quoted_post = Field()
    quoted_username = Field()
    user_total_posts = Field()
    user_posts_per_day = Field()
    post_subject = Field()
    img_urls = Field()
    other_urls = Field()
    embed_urls = Field()

