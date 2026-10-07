import string

import mlx.core as mx
import mlx.nn as nn

from ..gemma4 import Model as Gemma4Model
from ..jev.jev import Jev, merge_lora


class Model(Gemma4Model):
    decision_types = ("choice", "score", "bool", "noul")

    def __init__(self, config):
        super().__init__(config)
        ranges = config.decision_config["ranges"].values()
        self.head = nn.Linear(
            config.text_config.hidden_size, max(end for _, end in ranges)
        )

    def __call__(self, input_ids, mm_token_type_ids=None, **media):
        features = self.get_input_embeddings(input_ids=input_ids, **media)
        hidden = self.language_model.model(
            inputs_embeds=features.inputs_embeds,
            per_layer_inputs=features.per_layer_inputs,
            mm_token_type_ids=mm_token_type_ids,
        )
        logits = self.head(hidden[:, -1].astype(mx.float32))
        softcap = self.config.decision_config["softcap"]
        return softcap * mx.tanh(logits / softcap)

    def sanitize(self, weights):
        weights = merge_lora(weights, self.config.decision_config)
        head = {
            k: weights.pop(k) for k in list(weights) if k.startswith(("proj.", "head."))
        }
        head = {"head." + k.split(".", 1)[1]: v for k, v in head.items()}
        return {**super().sanitize(weights), **head}

    @property
    def quant_predicate(self):
        base = super().quant_predicate
        return lambda path, module: path != "head" and base(path, module)

    @property
    def cast_predicate(self):
        return lambda path: not path.startswith("head.")

    def predict(self, processor, state, questions, **kwargs):
        return Gev(self, processor).predict(state, questions, **kwargs)


class Gev(Jev):
    prefix = "<bos>"
    image_token = "<|image|>"

    def _labels(self):
        return string.ascii_uppercase[:16], None

    def _scores(self, kind, ids, media, count):
        start = self.settings["ranges"][kind][0]
        return self.model(ids, **media)[0, start : start + count]

    def _choice(self, state, images, question, options):
        if len(options) <= 16:
            return self._pass("choice", state, images, question, options)
        count = -(-len(options) // 16)
        size, extra = divmod(len(options), count)
        groups, start = [], 0
        for index in range(count):
            end = start + size + (index < extra)
            groups.append(list(range(start, end)))
            start = end
        in_group = {}
        for group in groups:
            scores = self._pass(
                "choice", state, images, question, [options[i] for i in group]
            )
            in_group.update(zip(group, scores))
        chosen = [max(group, key=lambda o: (in_group[o], -o)) for group in groups]
        rest = sorted(
            (o for group in groups for o in group if o not in set(chosen)),
            key=lambda o: (-in_group[o], o),
        )
        finalists = sorted(chosen + rest[: max(0, 16 - len(chosen))])
        final = dict(
            zip(
                finalists,
                self._pass(
                    "choice", state, images, question, [options[i] for i in finalists]
                ),
            )
        )
        group_of = {o: g for g, group in enumerate(groups) for o in group}
        share, cap = [0.0] * len(groups), [0.0] * len(groups)
        for o in finalists:
            share[group_of[o]] += final[o]
            cap[group_of[o]] += in_group[o]
        among = sum(a * b for a, b in zip(share, cap))
        p = [
            final[o] * among if o in final else share[group_of[o]] * in_group[o]
            for o in range(len(options))
        ]
        total = sum(p)
        return [value / total for value in p]
