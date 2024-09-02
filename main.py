import pandas as pd
import bm25s

from bm25s.hf import BM25HF


df = pd.read_csv("/home/ericyu8817/EHRNoteQA/data/mimiciv_new.csv")
corpus = list(df['new_text'])

#corpus = [
#    "a cat is a feline and likes to purr",
#    "a dog is the human's best friend and loves to play",
#    "a bird is a beautiful animal that can fly",
#    "a fish is a creature that lives in water and swims",
#]

retriever = BM25HF(corpus=corpus)
retriever.index(bm25s.tokenize(corpus))
user = "Eric0929"
retriever.save_to_hub(f"{user}/bm25s-mimiciv", include_readme=False)
