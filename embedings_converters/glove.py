import json
from typing import Dict, List

import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin


class GloveEmbedingsConverter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.__embedings = self.__load_embedings()
        self.__array = []
        return self

    def transform(self, X):
        self.__missing_counter = 0
        averaged_embedings = self.__convert_text_to_embdings(X["text"])
        embedings_df = pd.DataFrame(
            averaged_embedings, columns=list(range(1, 301))
        ).dropna(how="all")

        # print("missing counter is ", self.__missing_counter)
        # with open("temp.json", "w") as f:
        #     json.dump({"array": self.__array}, f)

        print("without saving out of bag vocabulary")

        return embedings_df

    def __convert_text_to_embdings(self, texts: pd.Series) -> list:
        embedings = []
        for text in texts:
            embedings_vector = []
            words = word_tokenize(text)
            for word in words:
                if self.__embedings.get(word) is not None:
                    embedings_vector.append(np.array(self.__embedings.get(word)))
                else:
                    embedings_vector.append(np.zeros(300))
                    self.__missing_counter += 1
                    self.__array.append(word)

            embedings_vector = np.mean(embedings_vector, axis=0)
            embedings.append(embedings_vector)
        return embedings

    def __load_embedings(self) -> Dict[str, np.ndarray]:
        embedings = {}
        with open("pretrained_models/glove.6B.300d.txt", encoding="utf-8") as f:
            for line in f:
                values = line.split(" ")
                if len(values):
                    word = values[0]
                    embedings_vector = np.array(values[1:], dtype=float)
                    embedings[word] = embedings_vector
        return embedings
