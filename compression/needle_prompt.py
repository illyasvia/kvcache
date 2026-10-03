from dataclasses import dataclass


PROMPT_MARKERS = (
    (
        "The document given to you by the user is",
        "\n\nNow, the question is:",
    ),
    (
        "用户现在给你的文档是",
        "\n\n现在请问：",
    ),
    (
        "The content of the long document is as follows\n\n<Document>",
        "\n</Document>\n\nBased on the information in the document, now please answer:",
    ),
    (
        "长文档的内容如下\n\n<文档>",
        "\n</文档>\n\n根据文档中的信息，现在请问：",
    ),
)


class PromptParseError(ValueError):
    """输入不符合 Needle prompt 的结构。"""


@dataclass(frozen=True)
class StructuredPrompt:
    instruction: str
    context: str
    question: str


def parse_needle_prompt(prompt: str) -> StructuredPrompt:
    """将现有 Needle benchmark 的单字符串 prompt 拆成三个语义区域。"""
    marker_pair = next(
        (
            (document_marker, question_marker)
            for document_marker, question_marker in PROMPT_MARKERS
            if document_marker in prompt and question_marker in prompt
        ),
        None,
    )
    if marker_pair is None:
        raise PromptParseError("无法识别 Needle prompt 的文档或问题边界标记")

    document_marker, question_marker = marker_pair
    document_pos = prompt.find(document_marker)
    question_pos = prompt.rfind(question_marker)
    context_start = document_pos + len(document_marker)
    if question_pos <= context_start:
        raise PromptParseError("问题边界位于文档边界之前，无法解析 prompt")

    instruction = prompt[:context_start].rstrip()
    context = prompt[context_start:question_pos].strip()
    question = prompt[question_pos:].strip()
    if not context:
        raise PromptParseError("文档内容为空")
    if not question:
        raise PromptParseError("问题内容为空")

    return StructuredPrompt(
        instruction=instruction,
        context=context,
        question=question,
    )
