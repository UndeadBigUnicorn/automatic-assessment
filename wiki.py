import wikipedia
import json


def find_between(s, first, last):
    """
    Find string between 2 substrings
    :param s: content, original string
    :param first: first substring
    :param last: second substring
    :return: found substring or empty string if nothing was found
    """
    try:
        start = s.index(first) + len(first)
        end = s.index(last, start)
        return s[start:end]
    except ValueError:
        return ""


def search_articles(key_phrase):
    return wikipedia.search(key_phrase)


def get_page(article):
    if len(article.strip()) == 0:
        return None
    try:
        return wikipedia.page(article)
    except wikipedia.exceptions.PageError:
        return None
    except wikipedia.exceptions.WikipediaException:
        return None



def get_see_also_links(content):
    # wikipedia articles that could be also parsed to get information about the category
    return find_between(content, "== See also ==", "== References ==").strip().splitlines()
