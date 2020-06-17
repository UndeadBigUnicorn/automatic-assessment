import scrapper
from nlp import NLP
from db import DB
import log
import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description="Automatic assessment for student works")

# Required positional argument
parser.add_argument("--category",
                    help="Required category name. Provide value in ''")

parser.add_argument("--file",
                    help="Filename of student work to check on")

args = parser.parse_args()

# workflow:
# 1. get all wikipedia articles that contains given category
# 2. use nlp in every article, find head words in sentences and highlight them
# 3. use nlp on wikipedia articles in depth 3 that are in "== See also ==" section on the original articles
# and make the same
# 4. save data in the db, rocksdb,  etc.
# 5. train entity recognition models on the saved datasets
# 6. use trained NER on the students works
# 7. check works for plagiarism (wikipedia texts and works between each other)

# category to search
category = args.category

# init db
db = DB(category)
log.info("Loaded database")

pages = db.get_pages()
if pages is None:
    pages = scrapper.look_for_articles(db, category)

log.debug("Articles: " + "; ".join(db.get_articles()))
log.info(f"Total Pages: {len(db.get_pages())}")

training_data = db.get_training_data()
if training_data is None:
    training_data = []
else:
    log.debug(f"Loaded training data from the database: {training_data}")

nlp = NLP(category, map(lambda page: page.content, pages), training_data)

log.info("NLP object was initialized")

if len(training_data) == 0:
    nlp.prepare_training_data()
    log.info("Prepared training data")
    db.save_training_data(nlp.training_data)
    log.info("Saved training data to the database")
    nlp.train_model()
    log.info("New model was trained")

with open(args.file) as f:
    doc = nlp.read_text(f.read())
    print([(ent.text, ent.label_) for ent in doc.ents])
