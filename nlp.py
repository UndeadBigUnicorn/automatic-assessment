import spacy
from spacy.util import minibatch, compounding
import warnings
import random

class NLP:
    def __init__(self, texts, training_data):
        # load spacy english pre-trained model
        self.nlp = spacy.load("en_core_web_sm")
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
            # mark chunks as new label which we looking for
            for chunk in doc.noun_chunks:
                train_entities.append((chunk.start_char, chunk.end_char, "TRAINED_CATEGORY"))

            # prevent overfitting and not forget old labels
            for ent in doc.ents:
                train_entities.append((ent.start_char, ent.end_char, ent.label_))

            self.training_data.append((doc.text, {"entities": train_entities}))

    def train_model(self, train_data, n_iter=100):
        ner = self.nlp.get_pipe("ner")

        for _, annotations in train_data:
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
                random.shuffle(train_data)
                losses = {}
                # batch up the examples using spaCy's minibatch
                batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
                for batch in batches:
                    texts, annotations = zip(*batch)
                    self.nlp.update(
                        texts,  # batch of texts
                        annotations,  # batch of annotations
                        drop=0.5,  # dropout - make it harder to memorise data
                        losses=losses,
                    )
                print("Losses", losses)

            # test the trained model
            for text, _ in train_data:
                doc = self.nlp(text)
                print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
                print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

            # save model to output directory

