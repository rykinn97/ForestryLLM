import unittest

from forestryllm.corpus import validate_records
from forestryllm.retrieval import lexical_search


class CorpusValidationTest(unittest.TestCase):
    def test_valid_record_passes(self):
        records = [
            {
                "chunk_id": "C1",
                "book_title": "Book",
                "chapter_title": "Chapter",
                "section_title": "Section",
                "page_start": 1,
                "page_end": 1,
                "chunk_title": "林木病害定义",
                "topic_type": "definition",
                "cleaned_text": "林木病害是用于测试的示例文本，长度足够通过短文本提示。",
                "keywords": "林木病害;定义",
                "citation_anchor": "Book-Chapter-Section-1-1",
            }
        ]
        result = validate_records(records, ["definition"])
        self.assertTrue(result.ok)
        self.assertEqual(result.summary["total_records"], 1)

    def test_duplicate_chunk_id_fails(self):
        record = {
            "chunk_id": "C1",
            "book_title": "Book",
            "chapter_title": "Chapter",
            "section_title": "Section",
            "page_start": 1,
            "page_end": 1,
            "chunk_title": "Title",
            "topic_type": "definition",
            "cleaned_text": "林木病害是用于测试的示例文本，长度足够通过短文本提示。",
            "keywords": "keyword",
            "citation_anchor": "anchor",
        }
        result = validate_records([record, dict(record)], ["definition"])
        self.assertFalse(result.ok)
        self.assertEqual(result.summary["duplicate_chunk_id_count"], 1)


class RetrievalTest(unittest.TestCase):
    def test_lexical_search_returns_relevant_chunk(self):
        records = [
            {"chunk_id": "C1", "chunk_title": "林木病害定义", "keywords": "林木病害", "cleaned_text": "林木病害定义文本"},
            {"chunk_id": "C2", "chunk_title": "遗传学对象", "keywords": "遗传学", "cleaned_text": "遗传学对象文本"},
        ]
        results = lexical_search("什么是林木病害", records, top_k=1)
        self.assertEqual(results[0]["chunk_id"], "C1")


if __name__ == "__main__":
    unittest.main()
