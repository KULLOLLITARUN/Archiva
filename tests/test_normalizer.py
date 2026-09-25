"""Tests for chatbot/normalizer.py — query normalization (Fix #19 whitelist)."""

from chatbot.normalizer import normalize


def test_strips_and_collapses_whitespace():
    assert normalize("  what   is  the policy?  ") == "what is the policy?"


def test_keeps_technical_punctuation():
    assert normalize("path: /usr/local/bin_v2 [error] {code=404}") == \
        "path: /usr/local/bin_v2 [error] {code=404}"


def test_keeps_non_ascii_letters():
    assert normalize("qu'est-ce que c'est déjà vu?") == "qu'est-ce que c'est déjà vu?"


def test_strips_disallowed_symbols():
    assert normalize("hello$world*foo") == "helloworldfoo"


def test_truncates_to_500_chars():
    long_query = "a" * 600
    result = normalize(long_query)
    assert len(result) == 500
