from learning_companion.cli import main
from learning_companion.knowledge import KnowledgeBase
from learning_companion.knowledge_cli import handle_knowledge_command


def test_cli_import_search_show_update_delete_without_chat_key(tmp_path, capsys):
    database = tmp_path / "knowledge.db"
    env_file = tmp_path / "settings.env"
    env_file.write_text(
        f"DATABASE_PATH={database.as_posix()}\nLLM_MODEL=\nLLM_API_KEY=\nKB_SEARCH_MODE=keyword\n",
        encoding="utf-8",
    )
    note = tmp_path / "有 空格.md"
    note.write_text("# 列表\n列表是可变对象。", encoding="utf-8")

    def run(*args):
        code = main(["--env-file", str(env_file), *args])
        return code, capsys.readouterr()

    code, output = run("--kb-import", str(note))
    assert code == 0 and "已导入" in output.out
    code, output = run("--kb-list")
    assert code == 0
    document_id = output.out.split()[0]
    code, output = run("--kb-search", "可变对象")
    assert code == 0 and "[KB:" in output.out and "列表是可变对象" in output.out
    code, output = run("--kb-show", document_id)
    assert code == 0 and "列表是可变对象" in output.out
    note.write_text("# 字典\n字典保存键值对。", encoding="utf-8")
    code, output = run("--kb-import", str(note))
    assert code == 0 and "已更新" in output.out
    code, output = run("--kb-search", "可变对象")
    assert code == 0 and "未找到" in output.out
    code, output = run("--kb-delete", document_id)
    assert code == 0 and note.exists()
    code, output = run("--kb-list")
    assert code == 0 and "暂无资料" in output.out


def test_chat_knowledge_command_accepts_quoted_windows_paths(settings, tmp_path, capsys):
    knowledge = KnowledgeBase(settings)
    path = tmp_path / "笔记 空格.md"
    path.write_text("# 递归\n递归需要终止条件。", encoding="utf-8")
    handle_knowledge_command(knowledge, f'import "{path}"')
    assert "已导入" in capsys.readouterr().out
    handle_knowledge_command(knowledge, "search 递归")
    assert "终止条件" in capsys.readouterr().out


def test_directory_import_recurses_and_reports_individual_failures(settings, tmp_path, capsys):
    from learning_companion.knowledge_cli import import_path

    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "valid.md").write_text("有效资料。", encoding="utf-8")
    nested = folder / "nested"
    nested.mkdir()
    (nested / "valid.txt").write_text("递归资料。", encoding="utf-8")
    knowledge = KnowledgeBase(settings)
    import_path(knowledge, folder)
    assert len(knowledge.documents()) == 2
    assert capsys.readouterr().out.count("已导入") == 2


def test_empty_cli_search_is_rejected_instead_of_starting_chat(tmp_path, capsys):
    env_file = tmp_path / "settings.env"
    env_file.write_text(f"DATABASE_PATH={(tmp_path / 'empty.db').as_posix()}\n", encoding="utf-8")
    assert main(["--env-file", str(env_file), "--kb-search", ""]) == 2
    assert "检索词" in capsys.readouterr().err
