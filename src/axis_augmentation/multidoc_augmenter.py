from itertools import permutations
from math import factorial
from random import sample
from typing import List

from datasets import load_dataset

from src.utils.constants import MultiDocConstants


class MultiDocAugmenter():
    """
    This augmenter is intended for multi-document tasks, and performs augmentation on the
    list of documents of each example in the dataset.
    """

    def add_random_contexts(self, docs: List[str], corpus: List[str],
                            n_new_docs: int = MultiDocConstants.DEFAULT_N_NEW_DOCS) -> List[str]:
        """
        Adds n_new_docs random contexts from the corpus to the end of the docs list.
        :param docs: a list of documents to augment
        :param corpus: a list of documents to sample from
        :param n_new_docs: the number of irrelevant documents to sample from the corpus
        :return: an augmented list of documents, where the original docs appear first,
        and n_new_docs irrelevant documents are added
        """
        irrelevant_docs = sample([doc for doc in corpus if doc not in docs], n_new_docs)
        augmented_docs = docs + irrelevant_docs
        return augmented_docs

    def permute_docs_order(self, docs: List[str], n_permutations: int = MultiDocConstants.DEFAULT_N_PERMUTATIONS) -> \
    list[list[str, ...]]:
        """
        Generates variations of the order of the documents in the list.
        :param docs: a list of documents to augment
        :param n_permutations: the number of permutations to generate
        :return: a list of min(n_permutations, len(docs)) tuples of docs, where each tuple is a permutation of the docs
        """
        # if example has one doc, no order augmentation is needed
        if len(docs) <= 1:
            return [docs]

        # generate all permutations of the docs
        n_iterations = min(n_permutations, factorial(len(docs)))
        augments = sample(list(permutations(docs)), n_iterations)
        return [list(item) for item in augments]

    def concatenate_docs(self, docs: List[str], concat_type: str = MultiDocConstants.SINGLE_DOC) -> str:
        """
        Concatenate the documents into a single string.
        :param docs: a list of documents to concatenate
        :param concat_type: the type of concatenation to perform. Choose from: ["single_doc", "2_newlines", "titles", "dashes"],
        or choose "special_<seperator>" where "seperator" is a specific string to use as a separator
        :return: a single string containing all documents concatenated
        """
        if concat_type == MultiDocConstants.SINGLE_DOC:
            # Add a single newline between documents
            return "\n".join(docs)

        elif concat_type == MultiDocConstants.DOUBLE_NEWLINES:
            # Add two newlines between documents
            return "\n\n".join(docs)

        elif concat_type == MultiDocConstants.TITLES:
            # Add titles to each document
            return "\n".join(
                [f"{MultiDocConstants.DOC_TITLE_FORMAT.format(i + 1)}\n{doc}\n" for i, doc in enumerate(docs)])

        elif concat_type == MultiDocConstants.DASHES:
            # Add dashes between documents
            return "\n".join(
                [f"{doc}\n{'-' * MultiDocConstants.DEFAULT_SEPARATOR_LENGTH}" for i, doc in enumerate(docs)])

        elif "special_" in concat_type:
            # Use the specified separator
            separator = concat_type.split("special_")[1]
            return separator.join(docs)

        else:
            raise ValueError(
                f"Invalid concat_type: {concat_type}. Choose from: ['{MultiDocConstants.SINGLE_DOC}', '{MultiDocConstants.DOUBLE_NEWLINES}', '{MultiDocConstants.TITLES}', '{MultiDocConstants.DASHES}'] or provide a specific string as a separator.")


if __name__ == "__main__":  # Example usage
    # Load the dataset (this is clapnq, a multi-document dataset intended for RAG)
    ds = load_dataset("PrimeQA/clapnq")['validation']['passages']
    docs = [ds[i][0]['text'] for i in range(3)]  # example 3 documents
    corpus = [item[0]['text'] for item in ds]  # entire corpus

    # run the augmenter on the example documents
    augmenter = MultiDocAugmenter()
    docs_extended = augmenter.add_random_contexts(docs, corpus, 2)
    docs_permutations = augmenter.permute_docs_order(docs_extended, 5)
    docs_concatenated = augmenter.concatenate_docs(docs_permutations[0], "titles")
    print(docs_concatenated)
