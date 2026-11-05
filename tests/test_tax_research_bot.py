from types import SimpleNamespace

from tax_research_bot import format_sources


def test_format_sources_handles_missing_metadata():
    documents = [
        SimpleNamespace(metadata={"source": "tax-guide.pdf", "page": 5}),
        SimpleNamespace(metadata={}),
    ]

    assert format_sources(documents) == [
        "- tax-guide.pdf (Page 5)",
        "- Unknown Source (Page N/A)",
    ]
