"""LLM merger разбивает объединённый пул на группы исходных кандидатов."""

import json
import re


def render_dedup_messages(title, paragraph_text, pool, prompt):
    data = {"title": title, "paragraph_text": paragraph_text,
            "candidates": [{"id": i, "triplet": t} for i, t in enumerate(pool)]}
    return [{"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(data, ensure_ascii=False)}]


def _unique_keys(items):
    values = dict(items)
    if len(values) != len(items):
        raise ValueError("Повторяющийся ключ JSON")
    return values


def merge_triplets(raw, pool):
    """Проверить полное разбиение, затем взять оригиналы представителей."""
    text = raw.strip()
    block = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", text, flags=re.I | re.S)
    result = json.loads(block[1] if block else text, object_pairs_hook=_unique_keys)
    if (not isinstance(result, dict) or set(result) != {"groups"}
            or not isinstance(result["groups"], list)):
        raise ValueError("Ожидался объект с единственным полем groups (список)")

    groups, seen = [], set()
    for group in result["groups"]:
        if not isinstance(group, dict) or set(group) != {"members", "representative"}:
            raise ValueError("Группа должна содержать members и representative")
        members, representative = group["members"], group["representative"]
        if (not isinstance(members, list) or not members
                or any(type(i) is not int or not 0 <= i < len(pool) for i in members)):
            raise ValueError("Группа должна содержать только существующие ID")
        if type(representative) is not int or representative not in members:
            raise ValueError("Представитель должен входить в свою группу")
        if len(set(members)) != len(members) or seen.intersection(members):
            raise ValueError("ID кандидата повторяется в разбиении")
        seen.update(members)
        groups.append({"members": sorted(members), "representative": representative})
    if len(seen) != len(pool):
        raise ValueError("Разбиение не учитывает все входные ID")

    groups.sort(key=lambda group: group["members"][0])
    return groups, [pool[group["representative"]] for group in groups]
