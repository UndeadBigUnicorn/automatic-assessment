import pickledb
import jsonpickle


class DB:
    def __init__(self, category):
        prefix = "_".join(category.split(" "))
        self.prefix = prefix + "_"
        self.db = pickledb.load('db', True)

    def set(self, key, value):
        self.set(self.prefix + key, value)

    def get(self, key):
        return self.get(self.prefix + key)

    def get_articles(self):
        try:
            return self.db.lgetall(self.prefix + "articles")
        except KeyError:
            return None

    def save_articles(self, articles):
        """
        Save searched articles in the db to not search them again
        :param articles: list of articles
        :return:
        """
        if self.get_articles() is None:
            self.db.lcreate(self.prefix + "articles")
        for article in articles:
            self.db.ladd(self.prefix + "articles", article)

    def get_pages(self):
        try:
            return list(map(lambda p: jsonpickle.decode(p), self.db.lgetall(self.prefix + "pages")))
        except KeyError:
            return None

    def get_page(self, page):
        if self.get_pages() is None:
            return None

        pages = list(filter(lambda p: p.title == page, self.get_pages()))
        if len(pages) == 0:
            return None
        return pages[0]

    def save_page(self, page):
        if self.get_pages() is None:
            self.db.lcreate(self.prefix + "pages")

        self.db.ladd(self.prefix + "pages", jsonpickle.encode(page))

    def get_training_data(self):
        try:
            return self.db.lgetall(self.prefix + "training_data")
        except KeyError:
            return None

    def save_training_data(self, data):
        if self.get_training_data() is None:
            self.db.lcreate(self.prefix + "training_data")

        self.db.lextend(self.prefix + "training_data", data)
