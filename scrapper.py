import wiki
import log


def scrap_articles(db, articles, read_articles, depth=1):
    """
    Save pages to the database recursively
    :param db: DB class
    :param articles: list of articles
    :param read_articles: list of already read articles to prevent duplications
    :param depth: max depth of scrapping
    :return: void
    """

    if depth >= 3:
        return

    for article in articles:
        if article in read_articles:
            continue

        # read wikipedia article page
        page = db.get_page(article)
        if page is None:
            page = wiki.get_page(article)
            if page is None:
                continue
            db.save_page(page)
            log.info(f"Saved wiki page {page.title} to the database")

        read_articles.append(article)

        # wikipedia 'See also' articles that could be also parsed to get information about the category
        scrap_articles(db, wiki.get_see_also_links(page.content), read_articles, depth+1)


def look_for_articles(db, category):
    """
    Populate wikipedia pages that contains given category to put them in NLP
    :param db: initialized db class
    :param category: category to look for
    :return: list of pages
    """

    # already read articles, it was made to prevent reading the same articles again
    read_articles = []
    # all articles
    articles = []

    # get all wikipedia articles that contains given category
    db_articles = db.get_articles()
    if db_articles is None:
        db_articles = wiki.search_articles(category)

    articles.extend(db_articles)

    scrap_articles(db, articles, read_articles)

    db.save_articles(read_articles)
    log.info("Saved populated wiki articles to the database")

    return db.get_pages()