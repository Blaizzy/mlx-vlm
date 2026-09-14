# Chat template fixtures

These full templates are public Apache-2.0 artifacts from the Qwen and Google
model repositories below. They remain under [Apache-2.0](LICENSE-APACHE-2.0.txt),
separately from this repository’s MIT-licensed code. All test messages are synthetic.
Standalone Jinja files are unchanged; Qwen2.5-VL is the unmodified decoded
`chat_template` string from its tokenizer configuration. No source NOTICE files
were available at these revisions.

| Fixture | Pinned source | Template SHA-256 |
| --- | --- | --- |
| qwen3_5.jinja | [Qwen/Qwen3.5-27B @ feea018b31f8](https://huggingface.co/Qwen/Qwen3.5-27B/blob/feea018b31f89dc0950e61da42577a7a4ab09169/chat_template.jinja) | `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` |
| qwen3_8.jinja | [Qwen/Qwen3.8-27B @ 412f8b6bd7f5](https://huggingface.co/Qwen/Qwen3.8-27B/blob/412f8b6bd7f5e922bea27645dcc3c80246b7f6a8/chat_template.jinja) | `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` |
| gemma4.jinja | [google/gemma-4-E2B-it @ 899537982545](https://huggingface.co/google/gemma-4-E2B-it/blob/899537982545a3e55ce64d34462b2efa5af85232/chat_template.jinja) | `0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5` |
| qwen2_5_vl.jinja | [Qwen/Qwen2.5-VL-7B-Instruct @ cc594898137f](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/cc594898137f460bfe9f0759e9844b3ce807cfb5/tokenizer_config.json) | `a0bc6f6fc7a29a80017a433e8f03a1cc1236e838a944a2d034295a60c4f2fddb` |

License labels are recorded in each repository’s README at the same revision.
Gemma4 also links [Google’s Apache-2.0 license](https://ai.google.dev/gemma/apache_2).
Qwen3.5’s embedded tokenizer-config template differs from its standalone Jinja;
these tests deliberately use the standalone file. Qwen3.5 and Qwen3.8 templates
are distinct, even though both checkpoints route through `qwen3_5`.

Tests use the Transformers Jinja renderer with these actual templates, without
weights, network access, or model-specific processors. They establish normalization
and representative rendering behavior, not universal checkpoint compatibility.
Qwen3.5 extracts inline thinking only when structured reasoning is not a string;
Qwen3.8 keeps inline markup as content and retains historical reasoning by default.
Gemma4 uses its own thought-channel and history rules. These policies are unchanged.
