"""多模态问答数据集使用的轻量评分函数。"""

import re
import string
import unicodedata
from collections.abc import Sequence


def normalize_answer(value) -> str:
    """统一大小写、Unicode、标点和空白，供开放式问答比较。"""
    text = unicodedata.normalize("NFKC", str(value or "")).lower().strip()
    text = text.translate(str.maketrans({char: " " for char in string.punctuation}))
    return " ".join(text.split())


def exact_match_score(prediction: str, references: Sequence[str]) -> float:
    normalized_prediction = normalize_answer(prediction)
    return 100.0 * any(
        normalized_prediction == normalize_answer(reference)
        for reference in references
    )


def _edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


def normalize_anls_answer(value) -> str:
    """按 DocVQA 评测约定仅统一大小写和连续空白。"""
    text = unicodedata.normalize("NFKC", str(value or "")).lower().strip()
    return " ".join(text.split())


def anls_score(prediction: str, references: Sequence[str]) -> float:
    """计算 DocVQA 的 ANLS，归一化编辑相似度低于 0.5 时记为 0。"""
    prediction = normalize_anls_answer(prediction)
    best_similarity = 0.0
    for reference in references:
        reference = normalize_anls_answer(reference)
        denominator = max(len(prediction), len(reference))
        similarity = 1.0 if denominator == 0 else (
            1.0 - _edit_distance(prediction, reference) / denominator
        )
        best_similarity = max(best_similarity, similarity)
    return 100.0 * best_similarity if best_similarity >= 0.5 else 0.0


def vqa_score(prediction: str, references: Sequence[str]) -> float:
    """计算 TextVQA 常用的 VQA 共识分数。"""
    prediction = normalize_answer(prediction)
    matches = sum(prediction == normalize_answer(reference) for reference in references)
    return 100.0 * min(matches / 3.0, 1.0)


def extract_choice(value: str) -> str | None:
    text = unicodedata.normalize("NFKC", str(value or "")).upper().strip()
    patterns = (
        r"(?:FINAL\s+ANSWER|ANSWER|OPTION|CHOICE)\s*(?:IS|:)?\s*\(?([A-Z])\)?",
        r"^\s*\(?([A-Z])\)?(?:[\s.):]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def multiple_choice_score(prediction: str, references: Sequence[str]) -> float:
    predicted_choice = extract_choice(prediction)
    reference_choices = {extract_choice(reference) for reference in references}
    reference_choices.discard(None)
    if predicted_choice is not None and reference_choices:
        return 100.0 * (predicted_choice in reference_choices)
    return exact_match_score(prediction, references)


SCORERS = {
    "anls": anls_score,
    "exact_match": exact_match_score,
    "multiple_choice": multiple_choice_score,
    "vqa": vqa_score,
}


def score_prediction(prediction: str, references: Sequence[str], scorer: str) -> float:
    if scorer not in SCORERS:
        raise ValueError(f"未知评分器: {scorer}")
    if not references:
        raise ValueError("参考答案不能为空")
    return SCORERS[scorer](prediction, references)
