import json
import sqlite3
from dataclasses import replace

import pytest
from langchain_core.embeddings import Embeddings

from learning_companion.knowledge import (
    TOOL_RESULT_CHARS,
    KnowledgeBase,
    KnowledgeBaseError,
    normalize_vector,
    split_document,
)


class ExampleEmbeddings(Embeddings):
    fail = False

    def embed_documents(self, texts):
        if self.fail:
            raise RuntimeError("provider error containing test-secret")
        return [[1.0, 0.0] if "共享" in text else [0.0, 1.0] for text in texts]

    def embed_query(self, text):
        return [1.0, 0.0] if text == "aliasing" else [0.0, 1.0]


@pytest.fixture
def note(tmp_path):
    path = tmp_path / "Python 笔记.md"
    path.write_text(
        "# Python 笔记\n\n## 浅拷贝\n浅拷贝创建新的外层容器，内部嵌套对象仍被共享。\n"
        "\n## 深拷贝\ndeepcopy 递归复制嵌套对象。\n",
        encoding="utf-8",
    )
    return path


def test_import_search_and_reopen_have_correct_source_and_lines(settings, note):
    knowledge = KnowledgeBase(settings)
    imported = knowledge.import_file(note)
    assert imported.status == "imported"
    assert imported.revision == 1
    knowledge = KnowledgeBase(settings)
    result = knowledge.search("浅拷贝")
    hit = result["results"][0]
    assert hit["document_id"] == imported.document_id
    assert hit["filename"] == note.name
    assert hit["citation"] == f"[KB:{imported.document_id}:r1:c1]"
    assert "共享" in hit["content"]
    assert hit["start_line"] == 1 and hit["end_line"] == 7
    assert knowledge.search("completely_absent_needle")["results"] == []


def test_duplicate_import_does_not_duplicate_index(settings, note):
    knowledge = KnowledgeBase(settings)
    first = knowledge.import_file(note)
    second = knowledge.import_file(note)
    assert second.status == "unchanged"
    assert second.document_id == first.document_id
    assert len(knowledge.documents()) == 1
    with knowledge.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM knowledge_fts").fetchone()[0] == first.chunks


def test_update_replaces_old_chunks_and_preserves_document_id(settings, note):
    knowledge = KnowledgeBase(settings)
    first = knowledge.import_file(note)
    note.write_text("# 闭包\n闭包保留外部作用域变量。\n", encoding="utf-8")
    second = knowledge.import_file(note)
    assert second.status == "updated"
    assert second.document_id == first.document_id and second.revision == 2
    assert knowledge.search("浅拷贝")["results"] == []
    assert knowledge.search("闭包")["results"][0]["revision"] == 2
    with knowledge.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM knowledge_fts").fetchone()[0] == second.chunks


def test_delete_cleans_index_and_keeps_source_file(settings, note):
    knowledge = KnowledgeBase(settings)
    imported = knowledge.import_file(note)
    knowledge.delete(imported.document_id)
    assert note.exists()
    assert knowledge.documents() == []
    assert knowledge.search("浅拷贝")["results"] == []
    with knowledge.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM knowledge_chunks").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM knowledge_fts").fetchone()[0] == 0
    with pytest.raises(KnowledgeBaseError, match="找不到"):
        knowledge.delete(imported.document_id)


@pytest.mark.parametrize("suffix,content", [(".env", "secret"), (".txt", "  "), (".txt", "a\x00b")])
def test_invalid_files_do_not_create_documents(settings, tmp_path, suffix, content):
    path = tmp_path / ("invalid" + suffix)
    path.write_text(content, encoding="utf-8")
    knowledge = KnowledgeBase(settings)
    with pytest.raises(KnowledgeBaseError):
        knowledge.import_file(path)
    assert knowledge.documents() == []


def test_gb18030_text_and_utf8_bom_are_supported(settings, tmp_path):
    knowledge = KnowledgeBase(settings)
    for encoding in ("gb18030", "utf-8-sig"):
        path = tmp_path / (encoding + ".txt")
        path.write_bytes("递归需要终止条件。\r\n".encode(encoding))
        knowledge.import_file(path)
    assert len(knowledge.search("递归")["results"]) == 2


def test_splitter_preserves_long_lines_and_line_locations():
    text = "# 章节\n\n" + "很长的内容" * 80 + "\n最后一行\n"
    chunks = split_document(text, 64, 12)
    assert all(len(chunk.content) <= 64 for chunk in chunks)
    assert chunks[0].start_line == 1
    assert chunks[-1].end_line == 4
    assert chunks[-1].section == "章节"
    for chunk in chunks:
        source_lines = "\n".join(text.splitlines()[chunk.start_line - 1 : chunk.end_line])
        assert chunk.content in source_lines


def test_markdown_headings_inside_code_fence_do_not_change_section():
    text = "# 正确章节\n```python\n# 不是章节\nprint('hi')\n```\n" + "内容" * 100
    chunks = split_document(text, 60, 5)
    assert all(chunk.section == "正确章节" for chunk in chunks)


def test_hybrid_can_retrieve_semantic_only_match_and_delete_it(settings, note):
    settings = replace(settings, knowledge_search_mode="hybrid", embedding_model="test-embedding")
    knowledge = KnowledgeBase(settings, ExampleEmbeddings())
    imported = knowledge.import_file(note)
    hit = knowledge.search("aliasing")["results"][0]
    assert hit["document_id"] == imported.document_id
    assert "共享" in hit["content"]
    knowledge.delete(imported.document_id)
    assert knowledge.search("aliasing")["results"] == []


def test_embedding_failure_during_update_leaves_old_index_intact(settings, note):
    settings = replace(settings, knowledge_search_mode="hybrid", embedding_model="test-embedding")
    embeddings = ExampleEmbeddings()
    knowledge = KnowledgeBase(settings, embeddings)
    imported = knowledge.import_file(note)
    note.write_text("完全不同的新内容", encoding="utf-8")
    embeddings.fail = True
    with pytest.raises(KnowledgeBaseError, match="原索引未修改") as failure:
        knowledge.import_file(note)
    assert "test-secret" not in str(failure.value)
    assert knowledge.document(imported.document_id)["revision"] == 1
    assert "共享" in knowledge.document(imported.document_id)["content"]
    embeddings.fail = False
    assert knowledge.search("aliasing")["results"]


def test_hybrid_requires_independent_embedding_configuration(settings, note):
    settings = replace(settings, knowledge_search_mode="hybrid", api_key="chat-key-not-reused")
    knowledge = KnowledgeBase(settings)
    with pytest.raises(KnowledgeBaseError, match="EMBEDDING_API_KEY"):
        knowledge.import_file(note)
    assert knowledge.documents() == []


def test_switching_to_hybrid_reindexes_existing_documents(settings, note):
    first = KnowledgeBase(settings).import_file(note)
    settings = replace(settings, knowledge_search_mode="hybrid", embedding_model="test-embedding")
    knowledge = KnowledgeBase(settings, ExampleEmbeddings())
    before = knowledge.search("aliasing")
    assert before["results"] == [] and before["unindexed_chunks"] > 0
    second = knowledge.import_file(note)
    assert second.document_id == first.document_id and second.revision == 2
    assert knowledge.search("aliasing")["unindexed_chunks"] == 0


def test_search_bounds_and_sql_special_characters(settings, note):
    knowledge = KnowledgeBase(settings)
    knowledge.import_file(note)
    for query, limit in (("", 3), ("x" * 257, 3), ("Python", 0), ("Python", 6)):
        with pytest.raises(KnowledgeBaseError):
            knowledge.search(query, limit)
    knowledge.search('"; DROP TABLE knowledge_documents; --')
    assert len(knowledge.documents()) == 1


def test_tool_payload_is_bounded(settings, tmp_path):
    settings = replace(settings, knowledge_chunk_size=2000, knowledge_chunk_overlap=100)
    knowledge = KnowledgeBase(settings)
    for index in range(5):
        path = tmp_path / f"资料{index}.md"
        path.write_text("# 学习材料\n" + "浅拷贝与共享嵌套对象。" * 140, encoding="utf-8")
        knowledge.import_file(path)
    payload = knowledge.tool_search("浅拷贝", 5)
    parsed = json.loads(payload)
    assert len(payload) <= TOOL_RESULT_CHARS
    assert parsed["results"] and parsed["omitted_results"]


@pytest.mark.parametrize("vector", [[], [0, 0], [float("nan"), 1], [float("inf"), 1]])
def test_invalid_embedding_vectors_are_rejected(vector):
    with pytest.raises(KnowledgeBaseError):
        normalize_vector(vector)


def test_database_failure_mid_update_rolls_back_text_chunks_and_fts(settings, note):
    knowledge = KnowledgeBase(settings)
    imported = knowledge.import_file(note)
    with knowledge.connection() as conn:
        conn.executescript("""
            CREATE TRIGGER fail_new_chunk BEFORE INSERT ON knowledge_chunks
            WHEN new.content LIKE '%FAIL_UPDATE%' BEGIN
                SELECT RAISE(ABORT, 'simulated failure');
            END;
        """)
    note.write_text("FAIL_UPDATE 新资料", encoding="utf-8")
    with pytest.raises(sqlite3.IntegrityError):
        knowledge.import_file(note)
    assert knowledge.document(imported.document_id)["revision"] == 1
    assert knowledge.search("浅拷贝")["results"]
    assert knowledge.search("FAIL_UPDATE")["results"] == []


def test_changed_embedding_model_requires_and_accepts_reimport(settings, note):
    settings = replace(settings, knowledge_search_mode="hybrid", embedding_model="model-a")
    imported = KnowledgeBase(settings, ExampleEmbeddings()).import_file(note)
    changed = KnowledgeBase(replace(settings, embedding_model="model-b"), ExampleEmbeddings())
    assert changed.search("aliasing")["results"] == []
    assert changed.search("aliasing")["unindexed_chunks"] > 0
    updated = changed.import_file(note)
    assert updated.document_id == imported.document_id and updated.revision == 2
    assert changed.search("aliasing")["results"]
