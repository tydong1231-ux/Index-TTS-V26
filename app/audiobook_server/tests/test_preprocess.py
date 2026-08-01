from audiobook_server.preprocess import normalize_coverage, repair_missing_fragments, split_novel_text


def test_split_novel_text_respects_maximum():
    text = "第一段。" * 300 + "\n\n" + "第二段。" * 180
    chunks = split_novel_text(text, target_chars=300, max_chars=420)
    assert len(chunks) > 2
    assert all(len(chunk) <= 420 for chunk in chunks)
    assert normalize_coverage("".join(chunks)) == normalize_coverage(text)


def test_repair_missing_narration_fragment():
    source = "林默走进站台。\n“你来了。”周岚说。"
    segments = [
        {"speaker": "旁白", "text": "林默走进站台。", "emotion": "平静", "confidence": 99, "needs_review": False},
        {"speaker": "周岚", "text": "你来了。", "emotion": "平静", "confidence": 95, "needs_review": False},
    ]
    repaired, missing = repair_missing_fragments(source, segments)
    assert missing
    assert any("周岚说" in segment["text"] for segment in repaired)
