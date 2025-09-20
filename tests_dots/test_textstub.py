from mlx_vlm.models.dots_ocr.tokenizer import SimpleTokenizer, render_chat


def test_simple_tokenizer_image_and_words():
    tokenizer = SimpleTokenizer(image_token_id=151652)
    ids = tokenizer.encode("hi there <image> now")
    assert 151652 in ids
    assert len(ids) == 4

    rendered = render_chat("hello <image>")
    assert "<image>" in rendered
