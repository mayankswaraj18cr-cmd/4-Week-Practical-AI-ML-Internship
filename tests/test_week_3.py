import numpy as np
import pytest

from week_3_nlp_transformers.text_preprocessing_embeddings import normalize_text, tfidf_embeddings, word_embedding_lookup


def test_text_features() -> None:
    assert normalize_text("Hello,   WORLD!") == "hello world"
    matrix, vectorizer = tfidf_embeddings(["cats are great", "dogs are great"])
    assert matrix.shape[0] == 2
    assert matrix.shape[1] == len(vectorizer.get_feature_names_out())
    assert word_embedding_lookup(["hello", "world"], 8).shape == (2, 8)


def test_attention_shape() -> None:
    torch = pytest.importorskip("torch")
    from week_3_nlp_transformers.transformer_attention_from_scratch import ScaledDotProductSelfAttention
    output, weights = ScaledDotProductSelfAttention(8, 2)(torch.randn(2, 4, 8))
    assert output.shape == (2, 4, 8)
    assert weights.shape == (2, 2, 4, 4)
