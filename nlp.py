import spacy
from spacy.util import minibatch, compounding
import warnings
import random
import log
from pathlib import Path


class NLP:
    def __init__(self, category, texts, training_data):
        self.category = "_".join(category.split(" "))
        models_dir = Path("models")
        if not models_dir.exists():
            models_dir.mkdir()
        category_dir = models_dir.joinpath(self.category)
        # load own category pre-trained model or spacy english pre-trained model
        self.nlp = spacy.load(category_dir if category_dir.exists() else "en_core_web_sm")
        self.texts = texts
        self.training_data = training_data

    def prepare_training_data(self):
        """
        Prepare data for NER model training
        :return: void
        """
        # use nlp on a list of texts
        for doc in self.nlp.pipe(self.texts):
            train_entities = []
            # hold already used chars for new category training to prevent overlapping with entities
            used_chunks = []
            # mark chunks as new label which we looking for
            for chunk in doc.noun_chunks:
                train_entities.append((chunk.start_char, chunk.end_char, "TRAINED_CATEGORY"))
                used_chunks.append({"start_char": chunk.start_char, "end_char": chunk.end_char})

            # TODO: improve speed
            def entity_is_used(ent):
                for chunk in used_chunks:
                    if chunk["start_char"] <= ent.start_char <= chunk["end_char"] \
                            or chunk["start_char"] <= ent.end_char <= chunk["end_char"] \
                            or ent.start_char <= chunk["start_char"] <= ent.end_char \
                            or ent.start_char <= chunk["end_char"] <= ent.end_char:
                        return True
                return False

            # prevent overfitting and not forget old labels
            for ent in doc.ents:
                # do not overlap tokens for trained category
                if not entity_is_used(ent):
                    train_entities.append((ent.start_char, ent.end_char, ent.label_))

            self.training_data.append((doc.text, {"entities": train_entities}))

    def train_model(self, n_iter=100):
        # create blank model
        # nlp = spacy.blank("en")
        # ner = nlp.create_pipe("ner")

        ner = self.nlp.get_pipe("ner")

        for _, annotations in self.training_data:
            for ent in annotations.get("entities"):
                ner.add_label(ent[2])

        # get names of other pipes to disable them during training
        pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe not in pipe_exceptions]

        # only train NER
        with self.nlp.disable_pipes(*other_pipes) and warnings.catch_warnings():
            # show warnings for misaligned entity spans once
            warnings.filterwarnings("once", category=UserWarning, module='spacy')

            # reset and initialize the weights randomly – but only if we're
            # training a new model
            self.nlp.begin_training()
            for itn in range(n_iter):
                random.shuffle(self.training_data)
                losses = {}
                # batch up the examples using spaCy's minibatch
                batches = minibatch(self.training_data, size=compounding(4.0, 32.0, 1.001))
                for batch in batches:
                    texts, annotations = zip(*batch)
                    self.nlp.update(
                        texts,  # batch of texts
                        annotations,  # batch of annotations
                        drop=0.5,  # dropout - make it harder to memorise data
                        losses=losses,
                    )
                log.debug("Losses", losses)

            # test the trained model
            for text, _ in self.training_data:
                doc = self.nlp(text)
                print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
                print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

            # save model to output directory
            output_dir = Path("models/" + self.category)
            if not output_dir.exists():
                output_dir.mkdir()
            self.nlp.to_disk(output_dir)
            log.debug(f"Saved model to {output_dir}")
            log.debug(f"Model for the category {self.category} was trained")

    def read_text(self, text):
        return self.nlp(text)
