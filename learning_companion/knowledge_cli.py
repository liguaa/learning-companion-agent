"""Knowledge management commands; independent of chat credentials."""

from pathlib import Path

from .knowledge import SUPPORTED_SUFFIXES, KnowledgeBase, KnowledgeBaseError


def import_path(knowledge: KnowledgeBase, path: Path) -> None:
    path = path.expanduser()
    if path.is_dir():
        root = path.resolve()
        files = sorted(
            item
            for item in root.rglob("*")
            if item.is_file()
            and item.suffix.lower() in SUPPORTED_SUFFIXES
            and item.resolve().is_relative_to(root)
        )
        if not files:
            raise KnowledgeBaseError("目录中没有可导入的 Markdown 或文本文件。")
    else:
        files = [path]
    failures = 0
    for file in files:
        try:
            result = knowledge.import_file(file)
            status = {"imported": "已导入", "updated": "已更新", "unchanged": "未变化"}[
                result.status
            ]
            print(
                f"{status}：{result.document_id}  {result.filename}  "
                f"r{result.revision}  {result.chunks} 个片段",
                flush=True,
            )
        except KnowledgeBaseError as error:
            failures += 1
            print(f"导入失败：{file.name}：{error}", flush=True)
    if failures:
        raise KnowledgeBaseError(f"{failures} 个文件导入失败；其他成功导入的文件已保存。")


def list_documents(knowledge: KnowledgeBase) -> None:
    documents = knowledge.documents()
    if not documents:
        print("知识库暂无资料，使用 --kb-import 或 /kb import 导入。")
    for document in documents:
        print(
            f"{document['id']}  {document['filename']}  r{document['revision']}  "
            f"{document['chunks']} 个片段\n  来源：{document['source_path']}"
        )


def show_document(knowledge: KnowledgeBase, document_id: str) -> None:
    document = knowledge.document(document_id)
    print(f"{document['filename']}  r{document['revision']}\n来源：{document['source_path']}\n")
    for number, line in enumerate(document["content"].splitlines(), 1):
        print(f"{number:>5}  {line}")


def search_documents(knowledge: KnowledgeBase, query: str) -> None:
    result = knowledge.search(query)
    if not result["results"]:
        print("未找到相关资料。可尝试资料中的关键词；本地全文检索不等同于语义理解。")
    for hit in result["results"]:
        section = f" · {hit['section']}" if hit["section"] else ""
        print(
            f"\n{hit['citation']} {hit['filename']}{section} "
            f"第 {hit['start_line']}～{hit['end_line']} 行\n{hit['content']}"
        )
    if result["unindexed_chunks"]:
        print(f"提示：{result['unindexed_chunks']} 个片段尚无当前模型的向量，请重新导入对应资料。")


def delete_document(knowledge: KnowledgeBase, document_id: str) -> None:
    knowledge.delete(document_id)
    print("资料及其检索索引已删除，原始文件保留。")


def handle_knowledge_command(knowledge: KnowledgeBase, command: str) -> None:
    parts = command.strip().split(maxsplit=1)
    action = parts[0] if parts else "list"
    argument = parts[1].strip() if len(parts) == 2 else ""
    if action == "list" and not argument:
        list_documents(knowledge)
    elif action == "import" and argument:
        if len(argument) >= 2 and argument[0] == argument[-1] and argument[0] in {'"', "'"}:
            argument = argument[1:-1]
        import_path(knowledge, Path(argument))
    elif action == "search" and argument:
        search_documents(knowledge, argument)
    elif action == "show" and argument:
        show_document(knowledge, argument)
    elif action == "delete" and argument:
        delete_document(knowledge, argument)
    else:
        raise KnowledgeBaseError(
            "用法：/kb list | import <路径> | search <关键词> | show <ID> | delete <ID>"
        )
