import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx

from mlx_vlm.speculative.drafters.qwen3_5_mtp.build_compact_head import (
    build_compact_proposal_head,
    select_vocab_ids,
)
from mlx_vlm.speculative.drafters.qwen3_5_mtp.compact_head import (
    CompactProposalHead,
    load_compact_proposal_head,
    tokenizer_vocab_sha256,
)
from mlx_vlm.speculative.drafters.qwen3_5_mtp.qwen3_5_mtp import Qwen3_5MTPDraftModel


class _FakeTokenizer:
    def __init__(self, vocab_size=96, added_ids=(90, 91)):
        self._vocab = {f"token-{index}": index for index in range(vocab_size)}
        self._added = {f"added-{index}": index for index in added_ids}
        self._vocab.update(self._added)

    def get_vocab(self):
        return dict(self._vocab)

    def get_added_vocab(self):
        return dict(self._added)


class CompactHeadTests(unittest.TestCase):
    def _source_checkpoint(self, root: Path):
        source = root / "source"
        source.mkdir()
        dense = mx.arange(96 * 64, dtype=mx.float32).reshape(96, 64) / 1000
        weight, scales, biases = mx.quantize(dense, group_size=32, bits=4)
        scales = scales.astype(mx.bfloat16)
        biases = biases.astype(mx.bfloat16)
        tensor_key = "language_model.lm_head.{}"
        tensors = {
            tensor_key.format("weight"): weight,
            tensor_key.format("scales"): scales,
            tensor_key.format("biases"): biases,
        }
        mx.save_safetensors(
            str(source / "model.safetensors"),
            tensors,
            metadata={"format": "mlx"},
        )
        config = {
            "model_type": "qwen3_5",
            "hidden_size": 64,
            "vocab_size": 96,
            "quantization": {"bits": 4, "group_size": 32, "mode": "affine"},
        }
        (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
        index = {"weight_map": {key: "model.safetensors" for key in tensors}}
        (source / "model.safetensors.index.json").write_text(
            json.dumps(index), encoding="utf-8"
        )
        return source, tensors

    def test_builder_copies_exact_rows_and_loads_with_target_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, tensors = self._source_checkpoint(root)
            tokenizer = _FakeTokenizer()
            output = build_compact_proposal_head(
                source,
                root / "compact",
                vocab_ids=[90, 1, 7],
                tokenizer=tokenizer,
            )
            head = load_compact_proposal_head(
                output,
                target_config={
                    "hidden_size": 64,
                    "vocab_size": 96,
                    "model_type": "qwen3_5",
                },
                tokenizer=tokenizer,
            )

            expected_ids = mx.array([1, 7, 90], dtype=mx.int32)
            self.assertEqual(head.vocab_ids.tolist(), expected_ids.tolist())
            for name in ("weight", "scales", "biases"):
                expected = mx.take(
                    tensors[f"language_model.lm_head.{name}"], expected_ids, axis=0
                )
                self.assertTrue(mx.array_equal(getattr(head, name), expected).item())
            self.assertTrue((output / "manifest.json").is_file())

    def test_builder_is_reproducible_and_supports_single_file_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = self._source_checkpoint(root)
            (source / "model.safetensors.index.json").unlink()
            tokenizer = _FakeTokenizer()
            outputs = []
            for name in ("first", "second"):
                output = build_compact_proposal_head(
                    source,
                    root / name,
                    vocab_ids=[1, 7, 90],
                    tokenizer=tokenizer,
                )
                outputs.append(output)

            self.assertEqual(
                (outputs[0] / "config.json").read_bytes(),
                (outputs[1] / "config.json").read_bytes(),
            )
            self.assertEqual(
                json.loads((outputs[0] / "manifest.json").read_text()),
                json.loads((outputs[1] / "manifest.json").read_text()),
            )

    def test_prefix_selection_includes_added_ids_and_alignment_rows(self):
        selected = select_vocab_ids(
            prefix_size=7,
            added_token_ids=[91, 90],
            vocab_size=96,
            row_multiple=8,
        )

        self.assertEqual(selected, list(range(14)) + [90, 91])

    def test_loader_rejects_changed_mapping_and_unknown_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = self._source_checkpoint(root)
            output = build_compact_proposal_head(
                source,
                root / "compact",
                vocab_ids=[1, 7, 90],
                tokenizer=_FakeTokenizer(),
            )
            config_path = output / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))

            config["selection"]["mapping_sha256"] = "0" * 64
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mapping hash"):
                load_compact_proposal_head(output)

            config["selection"]["mapping_sha256"] = None
            config["schema_version"] = 2
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema version"):
                load_compact_proposal_head(output)

    def test_builder_rejects_index_paths_outside_model_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = self._source_checkpoint(root)
            index_path = source / "model.safetensors.index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            first_key = next(iter(index["weight_map"]))
            index["weight_map"][first_key] = "../outside.safetensors"
            index_path.write_text(json.dumps(index), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "escapes"):
                build_compact_proposal_head(
                    source,
                    root / "compact",
                    vocab_ids=[1, 7, 90],
                    tokenizer=_FakeTokenizer(),
                )

    def test_builder_rejects_invalid_vocab_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = self._source_checkpoint(root)
            tokenizer = _FakeTokenizer()
            invalid = ([], [1, 1], [True], [1.5], [-1], [96])
            for index, values in enumerate(invalid):
                with self.subTest(values=values):
                    with self.assertRaises(ValueError):
                        build_compact_proposal_head(
                            source,
                            root / f"invalid-{index}",
                            vocab_ids=values,
                            tokenizer=tokenizer,
                        )

    def test_head_rejects_invalid_geometry_and_mapping(self):
        dense = mx.zeros((4, 64), dtype=mx.float32)
        weight, scales, biases = mx.quantize(dense, group_size=32, bits=4)
        cases = (
            {"vocab_ids": mx.array([0, 2, 1, 3])},
            {"vocab_ids": mx.array([0, 1, 1, 3])},
            {"vocab_ids": mx.array([0.0, 1.0, 2.0, 3.0])},
            {"scales": scales[:, :1]},
            {"biases": biases[:, :1]},
            {"weight": weight.astype(mx.int32)},
        )
        base = {
            "weight": weight,
            "scales": scales,
            "biases": biases,
            "vocab_ids": mx.arange(4, dtype=mx.int32),
            "group_size": 32,
            "bits": 4,
        }
        for change in cases:
            with self.subTest(change=tuple(change)):
                with self.assertRaises(ValueError):
                    CompactProposalHead(**(base | change))

    def test_target_validation_checks_shape_vocab_model_and_tokenizer(self):
        dense = mx.zeros((4, 64), dtype=mx.float32)
        weight, scales, biases = mx.quantize(dense, group_size=32, bits=4)
        tokenizer = _FakeTokenizer()
        head = CompactProposalHead(
            weight=weight,
            scales=scales,
            biases=biases,
            vocab_ids=mx.array([0, 1, 2, 90], dtype=mx.int32),
            group_size=32,
            bits=4,
            full_vocab_size=96,
            target_model_type="qwen3_5",
            target_tokenizer_sha256=tokenizer_vocab_sha256(tokenizer),
        )
        valid = {
            "hidden_size": 64,
            "vocab_size": 96,
            "model_type": "qwen3_5",
            "quantization": {"bits": 8, "group_size": 32, "mode": "affine"},
        }
        head.validate_target(valid, tokenizer=tokenizer)

        invalid = (
            ({**valid, "hidden_size": 32}, tokenizer),
            ({**valid, "vocab_size": 95}, tokenizer),
            ({**valid, "model_type": "qwen3_next"}, tokenizer),
            (valid, _FakeTokenizer(96, added_ids=(89, 91))),
        )
        for config, selected_tokenizer in invalid:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    head.validate_target(config, tokenizer=selected_tokenizer)

    def test_drafter_validates_attachment_and_can_restore_full_head(self):
        dense = mx.zeros((4, 64), dtype=mx.float32)
        weight, scales, biases = mx.quantize(dense, group_size=32, bits=4)
        head = CompactProposalHead(
            weight=weight,
            scales=scales,
            biases=biases,
            vocab_ids=mx.arange(4, dtype=mx.int32),
            group_size=32,
            bits=4,
            full_vocab_size=96,
            target_model_type="qwen3_5",
        )
        draft = SimpleNamespace(
            config=SimpleNamespace(
                text_config=SimpleNamespace(
                    hidden_size=64,
                    vocab_size=96,
                    model_type="qwen3_5",
                )
            )
        )

        Qwen3_5MTPDraftModel.set_compact_proposal_head(draft, head)
        self.assertIs(draft._compact_proposal_head, head)
        Qwen3_5MTPDraftModel.set_compact_proposal_head(draft, None)
        self.assertIsNone(draft._compact_proposal_head)
        with self.assertRaises(TypeError):
            Qwen3_5MTPDraftModel.set_compact_proposal_head(draft, object())

    def test_full_head_keeps_target_greedy_shortcut_after_compact_attachment(self):
        hidden = mx.zeros((1, 1, 64))
        greedy_token = Mock(return_value=mx.array([[11]]))
        full_head = Mock(return_value=mx.array([[[0.0, 1.0]]]))
        sampler = Mock(return_value=mx.array([[1]]))
        compact_head = SimpleNamespace(propose=Mock(return_value=mx.array([[7]])))
        draft = SimpleNamespace(
            _compact_proposal_head=None,
            _greedy_token=greedy_token,
            _lm_head_fn=full_head,
        )

        token = Qwen3_5MTPDraftModel._propose_token(draft, hidden, sampler, greedy=True)
        self.assertEqual(token.item(), 11)
        greedy_token.assert_called_once_with(hidden)
        full_head.assert_not_called()

        draft._compact_proposal_head = compact_head
        token = Qwen3_5MTPDraftModel._propose_token(draft, hidden, sampler, greedy=True)
        self.assertEqual(token.item(), 7)
        compact_head.propose.assert_called_once_with(hidden)

        token = Qwen3_5MTPDraftModel._propose_token(
            draft, hidden, sampler, greedy=False, compact_proposals=False
        )
        self.assertEqual(token.item(), 1)
        full_head.assert_called_once_with(hidden)
        sampler.assert_called_once()


if __name__ == "__main__":
    unittest.main()
