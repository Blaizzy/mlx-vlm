"""Grounded visual reasoning: Falcon Perception + local VLM on Apple Silicon.

Architecture
------------
User query + Image
        │
        ▼
 ┌──────────────────────────────────────────┐
 │  Orchestrator VLM  (any mlx-vlm model)  │
 │  Classifies task type, plans tool calls, │
 │  reasons over pixel-accurate metadata.   │
 └──────────────┬───────────────────────────┘
                │  tool calls
   ┌────────────┼──────────────────┬──────────────────────┐
   ▼            ▼                  ▼                      ▼
ground_expression  compute_relations  get_crop    spatial ops
(named slots)      (cross-slot IoU)   (appearance) (mask_ops.py)
 ┌──────────────┐  ┌──────────────┐               rank_by_x/y
 │ Falcon       │  │ IoU, centroid│               extreme_mask
 │ Perception   │  │ dist, ratio  │               nth_from
 └──────────────┘  └──────────────┘               exclude_extremes
   SoM image + JSON    JSON relations              compare_slot_positions
        │                                          closest_pair
        ▼
 Named mask slots:
   slot "A" → masks {1..N}
   slot "B" → masks {N+1..M}
   all_masks (merged, global IDs)

Usage
-----
    from agent import LocalVLMClient, run_agent, run_baseline
    from mlx_vlm import load

    fp_model, fp_processor = load("tiiuae/Falcon-Perception")
    vlm_client = LocalVLMClient("mlx-community/gemma-4-26b-a4b-it-4bit")

    result = run_agent(image, "Is the red player offside?",
                       fp_model, fp_processor, vlm_client)
    print(result.answer)
    result.final_image.show()
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
from PIL import Image

from mlx_vlm import load, generate
from fp_tools import compute_relations, masks_to_json, run_ground_expression
from mask_ops import SPATIAL_OPS, dispatch as spatial_dispatch
from viz import get_crop, render_final, render_som


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a visual reasoning agent. You answer questions about images \
by using Falcon Perception — a grounded segmentation model — that provides \
pixel-accurate measurements of every object you ask it to find.

You have full access to the image. Use what you see — colours, shapes, positions, \
context — to decide which objects to ground, which expressions to write, and which \
tools to call. Do not wait to be told what to look for: reason from the image itself.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT FALCON PERCEPTION GIVES YOU
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Calling ground_expression segments every matching object and returns:
  - A Set-of-Marks image (each object numbered and highlighted)
  - Exact JSON metadata for every mask:

      id            — unique integer (across all active slots)
      slot          — which expression produced this mask
      centroid_norm — pixel-accurate centre of mass, normalised 0–1
                      x=0 left · x=1 right · y=0 top · y=1 bottom
      area_fraction — fraction of the full image the mask covers
      bbox_norm     — bounding box (x1, y1, x2, y2), normalised 0–1
      image_region  — coarse label: "top-left", "center", etc.

These are pixel-level measurements — not visual estimates.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — LOOK AT THE IMAGE AND CLASSIFY THE TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Before calling any tool, look at the image and the query together.
Identify what objects are present, what colours/appearances they have,
and which task type fits the query:

  POSITIONAL / ORDERING
    The answer is in centroid_norm.x / centroid_norm.y.
    Use spatial operation tools — never sort numbers manually.

    Example queries → expressions → tools
    "Which duck is flying highest?"
      → ground("duck") → rank_by_y → the mask with rank 1 (lowest y) is highest

    "Which car is furthest to the right?"
      → ground("car") → extreme_mask(direction="rightmost") → answer

    "Which shelf is at the top?"
      → ground("shelf") → rank_by_y → rank 1 → answer

  CROSS-CATEGORY COMPARISON
    Ground each category into its own named slot, then compare.
    Look at the image to decide what colours/appearances define each category.

    Example queries → plan
    "Are there more red balls or blue balls?"
      → ground("red ball", slot="red") → ground("blue ball", slot="blue")
      → compare counts in JSON → answer

    "Which group of players is further forward?"
      → look at image to identify team colours (e.g. white vs dark jersey)
      → ground("white jersey player", slot="white")
      → ground("dark jersey player", slot="dark")
      → compare_slot_positions(axis="x" or "y" depending on direction of play)
      → answer

    "Which bird species is higher up in the image?"
      → identify the two species visually (e.g. mallard vs heron)
      → ground each into its own slot → rank_by_y each → compare top ranks

    "Is the boundary of group A past the boundary of group B?"
      → ground each group → exclude_extremes if outliers exist
      → extreme_mask(group A) vs nth_from(group B, n=1) → compare x values → answer

  COUNTING
    The number of masks returned by ground_expression IS the count.
    → ground → read n_masks from the JSON → answer

    Example: "How many people are wearing red?"
      → ground("person in red") → n_masks → answer

  VISUAL APPEARANCE
    You need fine detail not visible at full-image scale.
    → ground the object → get_crop → answer

    Example: "What does the sign say?"
      → ground("sign") → get_crop → read text from crop → answer

    Example: "What brand logo is on the shirt?"
      → ground("shirt") → get_crop → identify logo → answer

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOOLS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

── PERCEPTION ──────────────────────────────────────────────────────────────────

ground_expression  {"expression": "<noun phrase>", "slot": "<label>"}
  Segments all instances matching the expression and stores them in the named slot.
  Each slot is independent — a NEW slot name does NOT erase previous slots.
  Calling with the SAME slot name replaces that slot's masks.
  "slot" is optional and defaults to "default".
  Expression MUST describe VISUAL APPEARANCE only (colour, clothing, shape).
    Good: "red player"  "player in blue"  "yellow jacket"  "flying bird"  "white car"
    Bad:  "defender"  "attacker"  "winner"    ← semantic roles, not visual appearance
          "five red cars"  "leftmost duck"    ← counts or positions belong in tools
          "duck and goose"                    ← conjunctions: ground each separately

── SPATIAL OPERATIONS ──────────────────────────────────────────────────────────

rank_by_x  {"slot": "<label>"}
  Ranks all masks in the slot left→right (rank 1 = leftmost).

rank_by_y  {"slot": "<label>"}
  Ranks all masks in the slot top→bottom (rank 1 = topmost / highest in image).

extreme_mask  {"slot": "<label>", "direction": "leftmost|rightmost|topmost|bottommost"}
  Returns the single mask at the given positional extreme.

nth_from  {"slot": "<label>", "n": <int>, "direction": "left|right|top|bottom"}
  Returns the n-th mask sorted by direction (1-indexed from the extreme).
  n=1 is the extreme itself; n=2 is one step in from the extreme.

exclude_extremes  {"slot": "<label>", "axis": "x|y", "n": <int>}
  Removes the n masks from each end of the axis. Returns the remaining masks.
  Use to drop outlier detections before comparing boundaries.

filter_by_size  {"slot": "<label>", "top_n": <int>, "min_area": <float>, "max_area": <float>}
  Filters masks by area_fraction. All parameters are optional.
  Use to remove spurious tiny detections or keep only the N largest.

compare_slot_positions  {"slot_a": "<label>", "slot_b": "<label>", "axis": "x|y"}
  Compares positional distributions (mean/min/max) of two slots.
  Returns which slot is further right (x) or further down (y).

closest_pair  {"slot_a": "<label>", "slot_b": "<label>"}
  Returns the nearest pair of masks across two slots by centroid distance.

── APPEARANCE ───────────────────────────────────────────────────────────────────

get_crop  {"mask_id": <int>}
  Zoomed crop of one mask for fine-grained visual inspection.
  Use ONLY for appearance tasks (text, logos, subtle colours).

── RELATIONS ────────────────────────────────────────────────────────────────────

compute_relations  {"mask_ids": [id, id, ...]}
  Pairwise IoU, centroid distances, relative positions, and size ratios.
  IDs can span multiple slots.

── FINAL ANSWER ─────────────────────────────────────────────────────────────────

answer  {"response": "<text>", "supporting_mask_ids": [<ints>]}
  Call as soon as you have a confident conclusion.
  supporting_mask_ids lists the mask IDs that justify the answer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Every response MUST use this exact format:

<think>
Look at the image. Identify the objects and their visual properties.
Classify the task type. Name the slots and expressions you will use.
Plan the minimum tool calls needed to reach a confident answer.
</think>
<tool>{"name": "<tool_name>", "parameters": {<json>}}</tool>

Rules:
  1. Always include both <think>...</think> and <tool>...</tool>.
  2. Exactly ONE tool call per turn. Stop immediately after </tool>.
  3. No text after the </tool> closing tag.
  4. If ground_expression returns 0 masks, retry with a more visual expression
     (use colour or clothing — never abstract roles like "leader" or "winner").
  5. For positional queries, use spatial operation tools — never sort numbers manually.
  6. Call answer as soon as the results support a confident conclusion.
"""


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

_TOOL_RE = re.compile(r"<tool>(.*?)</tool>", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TraceStep:
    """A single step in the agent's reasoning trace."""

    step: int
    think: str = ""
    tool_name: str = ""
    tool_params: dict = field(default_factory=dict)
    result_text: str = ""
    som_image: Image.Image = None  # set when ground_expression returns a SoM


@dataclass
class GroundedReasoningResult:
    """Output of a completed agent run."""

    answer: str
    supporting_mask_ids: list = field(default_factory=list)
    final_image: Image.Image = None
    n_fp_calls: int = 0
    n_vlm_calls: int = 0
    trace: list = field(default_factory=list)  # list[TraceStep]


# ---------------------------------------------------------------------------
# Local VLM client
# ---------------------------------------------------------------------------

class LocalVLMClient:
    """Local VLM wrapper — works with any mlx-vlm compatible model.

    The orchestrator role can be filled by Gemma4, Qwen3-VL, or any other
    model supported by mlx-vlm.  Images stay as PIL objects throughout;
    no base64 encoding is needed.
    """

    def __init__(self, model_id, max_tokens=4096, temperature=0.0):
        print(f"Loading orchestrator VLM: {model_id} ...")
        self.model, self.processor = load(model_id)
        self.max_tokens = max_tokens
        self.temperature = temperature

    @staticmethod
    def _uniform_size(images):
        """Resize all images to match the first image's dimensions.

        Gemma4's aspect-ratio-preserving resize produces different output shapes
        for images of different sizes, causing mx.concatenate to fail when they
        are batched together.  Resizing to a common size before processing ensures
        the processor stacks them into a single array rather than a list of arrays.
        """
        if len(images) <= 1:
            return images
        target_w, target_h = images[0].size
        return [
            img if img.size == (target_w, target_h)
            else img.resize((target_w, target_h), Image.LANCZOS)
            for img in images
        ]

    def send(self, messages):
        """Send messages to the local VLM and return the response string."""
        vlm_messages = []
        all_images = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # Plain-string content (system message or simple assistant turn)
            if isinstance(content, str):
                vlm_messages.append({"role": role, "content": content})
                continue

            pil_imgs, texts = [], []
            for item in content:
                if item.get("type") == "pil_image":
                    pil_imgs.append(item["image"])
                elif item.get("type") == "text":
                    texts.append(item["text"])

            all_images.extend(pil_imgs)
            text = " ".join(texts)

            if role == "user" and pil_imgs:
                # Image tokens come first — matches Gemma4 / most VLM chat templates
                vlm_messages.append({
                    "role": "user",
                    "content": [{"type": "image"}] * len(pil_imgs)
                    + [{"type": "text", "text": text}],
                })
            elif role == "assistant":
                vlm_messages.append({"role": "assistant", "content": text})
            else:
                vlm_messages.append({"role": role, "content": text})

        prompt = self.processor.apply_chat_template(
            vlm_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        try:
            imgs_for_generate = self._uniform_size(all_images) if all_images else None
            result = generate(
                self.model,
                self.processor,
                prompt,
                imgs_for_generate,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                verbose=False,
            )
            mx.clear_cache()
            return result.text if hasattr(result, "text") else str(result)
        except Exception as e:
            print(f"[LocalVLMClient] Error: {e}")
            return None


# ---------------------------------------------------------------------------
# Tool-call parsing
# ---------------------------------------------------------------------------

def _parse_tool_call(text):
    """Extract and parse the JSON inside the first <tool>...</tool> block."""
    m = _TOOL_RE.search(text)
    if not m:
        return None
    raw = m.group(1).strip().replace("}}}", "}}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Context management
# ---------------------------------------------------------------------------

def _is_som_message(msg):
    """Return True if msg is a user message containing a SoM image."""
    if msg.get("role") != "user":
        return False
    content = msg.get("content", [])
    if not isinstance(content, list):
        return False
    return any(
        isinstance(item, dict)
        and item.get("type") == "pil_image"
        and item.get("is_som", False)
        for item in content
    )


def _prune_context(messages):
    """Keep message history compact.

    Strategy:
      - Always keep messages[0] (system) and messages[1] (original user image).
      - Strip the pixel data from all SoM messages except the most recent —
        the model only needs to see the current combined SoM, not the history.
        Older SoMs are replaced with a text placeholder so their metadata text
        stays in context.
      - Keep the last 12 tail messages.
    """
    if len(messages) <= 4:
        return messages

    head = messages[:2]
    tail = messages[2:]

    last_som_idx = -1
    for i, msg in enumerate(tail):
        if _is_som_message(msg):
            last_som_idx = i

    if last_som_idx == -1:
        return head + tail[-12:]

    # Replace pixel data in older SoM messages with a text placeholder
    cleaned = []
    for i, msg in enumerate(tail):
        if _is_som_message(msg) and i < last_som_idx:
            new_content = []
            for item in msg["content"]:
                if item.get("type") == "pil_image" and item.get("is_som"):
                    new_content.append({
                        "type": "text",
                        "text": "[Previous Set-of-Marks image — superseded by latest SoM]",
                    })
                else:
                    new_content.append(item)
            cleaned.append({**msg, "content": new_content})
        else:
            cleaned.append(msg)

    return head + cleaned[-12:]


# ---------------------------------------------------------------------------
# Baseline (no grounding)
# ---------------------------------------------------------------------------

def run_baseline(image, query, vlm_client):
    """Direct VLM answer with no Falcon Perception grounding.

    Uses the documented mlx-vlm apply_chat_template API to ensure image
    tokens are inserted correctly for every model (including Gemma4 which
    computes a variable number of soft tokens per image based on dimensions).

    Use alongside run_agent to measure how much grounded reasoning
    improves over a plain visual QA baseline.
    """
    from mlx_vlm.prompt_utils import apply_chat_template as mlx_apply_chat_template

    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB")

    prompt = mlx_apply_chat_template(
        vlm_client.processor,
        vlm_client.model.config,
        [
            {
                "role": "system",
                "content": (
                    "You are a helpful visual assistant. "
                    "Answer the user's question about the image concisely and accurately."
                ),
            },
            {"role": "user", "content": query},
        ],
        num_images=1,
    )

    result = generate(
        vlm_client.model,
        vlm_client.processor,
        prompt,
        image=[image],
        max_tokens=vlm_client.max_tokens,
        temperature=vlm_client.temperature,
        verbose=False,
    )
    mx.clear_cache()
    return result.text if hasattr(result, "text") else str(result)


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def run_agent(
    image,
    query,
    fp_model,
    fp_processor,
    vlm_client,
    max_steps=20,
    fp_max_tokens=512,
    auto_relations_max=6,
    max_corrections=2,
    verbose=True,
):
    """Run the grounded reasoning agent on image with query.

    Args:
        image:               PIL Image or path to image file.
        query:               Natural-language question or task.
        fp_model:            Loaded Falcon Perception model.
        fp_processor:        Matching processor for fp_model.
        vlm_client:          LocalVLMClient instance.
        max_steps:           Hard cap on VLM calls before raising.
        fp_max_tokens:       Token budget per Falcon Perception call.
        auto_relations_max:  Auto-append pairwise relations when a single
                             ground_expression returns at most this many
                             masks. Set to 0 to disable.
        max_corrections:     Max consecutive correction turns before falling
                             back to extracting the answer from the model's
                             last <think> block directly.
        verbose:             Print per-step summaries (think + tool) to stdout.

    Returns:
        GroundedReasoningResult.
    """
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    pil_image = image.convert("RGB")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "pil_image", "image": pil_image},
                {"type": "text", "text": f"User query: {query}"},
            ],
        },
    ]

    # Named mask slots: slot_name -> {global_id: meta}
    mask_slots = {}
    # Merged flat dict of all active masks with globally unique IDs
    all_masks = {}
    # Monotonically increasing global ID counter
    next_mask_id = 1

    n_fp_calls = 0
    n_vlm_calls = 0
    # Tracks consecutive correction turns so we can clean them up on success
    correction_depth = 0
    # Last <think> block seen — used as fallback answer when corrections exceed cap
    last_think_text = ""
    # Accumulates one TraceStep per tool call for export
    trace: list = []

    if verbose:
        print(f"\n{'─' * 60}")
        print(f"  Grounded Reasoning Agent")
        print(f"  Query: {query!r}")
        print(f"{'─' * 60}")

    for step in range(max_steps):

        if verbose:
            print(f"\n[Step {step + 1}] Calling VLM ...")

        response_text = vlm_client.send(messages)
        n_vlm_calls += 1

        if response_text is None:
            raise RuntimeError(
                f"VLM returned None at step {step + 1}. "
                "Check that the model loaded correctly."
            )

        # Always capture the latest <think> content for fallback extraction
        think_m = _THINK_RE.search(response_text)
        if think_m:
            last_think_text = think_m.group(1).strip()

        if verbose:
            if think_m:
                t = last_think_text
                print(f"  [think] {t[:400]}{'...' if len(t) > 400 else ''}")
            tool_m = _TOOL_RE.search(response_text)
            if tool_m:
                print(f"  [tool]  {tool_m.group(1).strip()[:200]}")

        # Parse tool call — retry with correction messages on failure
        tool_call = _parse_tool_call(response_text)
        if tool_call is None:
            correction_depth += 1

            # Fallback: extract answer from the last <think> block directly
            if correction_depth > max_corrections:
                if verbose:
                    print(
                        f"  [warn] Correction cap ({max_corrections}) reached. "
                        "Extracting answer from last <think> block."
                    )
                fallback_answer = last_think_text or response_text
                return GroundedReasoningResult(
                    answer=fallback_answer,
                    supporting_mask_ids=[],
                    final_image=pil_image.copy(),
                    n_fp_calls=n_fp_calls,
                    n_vlm_calls=n_vlm_calls,
                    trace=trace,
                )

            if verbose:
                print(f"  [warn] No <tool> tag (attempt {correction_depth}), sending correction ...")

            # On the first attempt, give a generic reminder.
            # On subsequent attempts, show an explicit fill-in-the-blank answer template
            # so the model understands it should call `answer` now.
            if correction_depth == 1:
                correction_text = (
                    "Your response did not contain a <tool> tag. "
                    "You MUST end every response with a tool call in this exact format:\n"
                    "<think>brief reasoning</think>\n"
                    '<tool>{"name": "<tool_name>", "parameters": {<json>}}</tool>'
                )
            else:
                correction_text = (
                    "Still no <tool> tag. You have already reasoned correctly — "
                    "now call the answer tool. Replace the placeholders:\n\n"
                    "<think>I have identified the answer from the metadata.</think>\n"
                    '<tool>{"name": "answer", "parameters": {'
                    '"response": "<your conclusion in one sentence>", '
                    '"supporting_mask_ids": [<comma-separated mask IDs>]}}</tool>'
                )

            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": response_text}],
            })
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": correction_text}],
            })
            # Prune context during correction loops to prevent unbounded growth
            messages = _prune_context(messages)
            continue

        # Successful parse — remove any pending correction messages from context
        if correction_depth > 0:
            messages = messages[:len(messages) - 2 * correction_depth]
            correction_depth = 0

        tool_name = tool_call.get("name", "")
        params = tool_call.get("parameters", {})

        # Start a trace entry for this step
        _ts = TraceStep(
            step=step + 1,
            think=last_think_text,
            tool_name=tool_name,
            tool_params=params,
        )

        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": response_text}],
        })

        # --- ground_expression ---
        if tool_name == "ground_expression":
            expression = params.get("expression", "")
            slot = params.get("slot", "default")
            if verbose:
                print(f"  → ground_expression({expression!r}, slot={slot!r})")

            raw_masks = run_ground_expression(
                fp_model, fp_processor, pil_image, expression,
                max_new_tokens=fp_max_tokens,
            )
            n_fp_calls += 1

            # Assign globally unique IDs and tag each mask with its slot
            new_slot_masks = {}
            for local_id in sorted(raw_masks.keys()):
                meta = raw_masks[local_id].copy()
                meta["id"] = next_mask_id
                meta["slot"] = slot
                new_slot_masks[next_mask_id] = meta
                next_mask_id += 1

            # Update slot (replaces if same slot name called again)
            mask_slots[slot] = new_slot_masks

            # Rebuild flat merged dict from all slots
            all_masks = {}
            for s_masks in mask_slots.values():
                all_masks.update(s_masks)

            n_new = len(new_slot_masks)
            n_total = len(all_masks)

            if verbose:
                print(
                    f"     → {n_new} new mask(s) in slot '{slot}' "
                    f"| {n_total} total active across {len(mask_slots)} slot(s)"
                )

            if n_new == 0:
                tool_result = [{
                    "type": "text",
                    "text": (
                        f"ground_expression({expression!r}) returned 0 masks. "
                        "Try a more general expression."
                    ),
                }]
            else:
                # Render combined SoM of ALL active slots
                som_image = render_som(pil_image, all_masks)
                meta_json = json.dumps(
                    {"n_masks": n_total, "masks": masks_to_json(all_masks)},
                    indent=2,
                )
                tool_result = [
                    {"type": "pil_image", "image": som_image, "is_som": True},
                    {
                        "type": "text",
                        "text": (
                            f"ground_expression(slot='{slot}') → {n_new} new mask(s). "
                            f"Total active: {n_total} mask(s) across slots "
                            f"{list(mask_slots.keys())}.\n\n"
                            f"Mask metadata:\n{meta_json}"
                        ),
                    },
                ]

                # Auto-compute pairwise relations for small new batches
                new_ids = list(new_slot_masks.keys())
                if 2 <= len(new_ids) <= auto_relations_max:
                    relations = compute_relations(all_masks, new_ids)
                    tool_result.append({
                        "type": "text",
                        "text": (
                            f"Pairwise relations (auto-computed for "
                            f"{len(new_ids)} new masks in slot '{slot}'):\n"
                            + json.dumps(relations, indent=2)
                        ),
                    })

            messages.append({"role": "user", "content": tool_result})

            # Record trace step
            if n_new == 0:
                _ts.result_text = f"ground_expression({expression!r}) returned 0 masks."
            else:
                _ts.som_image = som_image
                _ts.result_text = (
                    f"ground_expression(slot='{slot}') → {n_new} new mask(s). "
                    f"Total: {len(all_masks)} across {list(mask_slots.keys())}."
                )
            trace.append(_ts)

        # --- get_crop ---
        elif tool_name == "get_crop":
            mask_id = int(params.get("mask_id", -1))
            if verbose:
                print(f"  → get_crop(mask_id={mask_id})")

            if mask_id not in all_masks:
                _crop_text = (
                    f"get_crop failed: mask_id={mask_id} not found. "
                    f"Active IDs: {sorted(all_masks.keys())}"
                )
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": _crop_text}],
                })
                _ts.result_text = _crop_text
            else:
                crop_img = get_crop(pil_image, all_masks[mask_id])
                slot_label = all_masks[mask_id].get("slot", "?")
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "pil_image", "image": crop_img},
                        {"type": "text", "text": f"Zoomed crop of mask {mask_id} (slot: {slot_label})."},
                    ],
                })
                _ts.result_text = f"Zoomed crop of mask {mask_id} (slot: {slot_label})."
            trace.append(_ts)

        # --- spatial operations (mask_ops.py) ---
        elif tool_name in SPATIAL_OPS:
            if verbose:
                print(f"  → {tool_name}({', '.join(f'{k}={v!r}' for k, v in params.items())})")
            try:
                result = spatial_dispatch(tool_name, all_masks, params)
                result_text = f"{tool_name} result:\n{json.dumps(result, indent=2)}"
            except (KeyError, IndexError, ValueError) as exc:
                result_text = f"{tool_name} error: {exc}"
                if verbose:
                    print(f"  [warn] {result_text}")

            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": result_text}],
            })
            _ts.result_text = result_text
            trace.append(_ts)

        # --- compute_relations ---
        elif tool_name == "compute_relations":
            mask_ids = params.get("mask_ids", [])
            if verbose:
                print(f"  → compute_relations(mask_ids={mask_ids})")

            relations = compute_relations(all_masks, mask_ids)
            rel_text = f"compute_relations result:\n{json.dumps(relations, indent=2)}"
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": rel_text}],
            })
            _ts.result_text = rel_text
            trace.append(_ts)

        # --- answer ---
        elif tool_name == "answer":
            response_final = params.get("response", "")
            selected_ids = [int(i) for i in params.get("supporting_mask_ids", [])]

            if verbose:
                print(f"\n{'─' * 60}")
                print(f"  Answer: {response_final}")
                print(f"  Support masks: {selected_ids}")
                print(f"  FP calls: {n_fp_calls}  |  VLM calls: {n_vlm_calls}")
                print(f"{'─' * 60}\n")

            final_image = (
                render_final(pil_image, all_masks, selected_ids)
                if selected_ids and all_masks
                else pil_image.copy()
            )

            _ts.result_text = response_final
            trace.append(_ts)

            return GroundedReasoningResult(
                answer=response_final,
                supporting_mask_ids=selected_ids,
                final_image=final_image,
                n_fp_calls=n_fp_calls,
                n_vlm_calls=n_vlm_calls,
                trace=trace,
            )

        else:
            _known = (
                "ground_expression, answer, get_crop, compute_relations, "
                + ", ".join(sorted(SPATIAL_OPS.keys()))
            )
            raise ValueError(
                f"Unknown tool '{tool_name}' at step {step + 1}. "
                f"Expected one of: {_known}"
            )

        messages = _prune_context(messages)

    raise RuntimeError(
        f"Agent exceeded max_steps={max_steps} without calling 'answer'."
    )


# ---------------------------------------------------------------------------
# Results export
# ---------------------------------------------------------------------------

def save_run_results(
    result: GroundedReasoningResult,
    baseline_answer: str,
    query: str,
    original_image: Image.Image,
    output_dir: str,
    run_name: str = "run",
) -> str:
    """Persist a grounded-reasoning run to disk.

    Layout::

        <output_dir>/
            images/
                <run_name>_original.png
                <run_name>_final.png
                <run_name>_step<N>_som.png   (one per ground_expression call)
            <run_name>.json

    The JSON contains all text fields and relative paths to the images so the
    whole ``output_dir`` folder is self-contained and portable.

    Returns the path to the written JSON file.
    """
    out = Path(output_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    def _save_img(img: Image.Image, name: str) -> str:
        """Save PIL image, return path relative to output_dir."""
        path = img_dir / name
        img.save(path)
        return str(path.relative_to(out))

    # ── core images ───────────────────────────────────────────────────────────
    original_path = _save_img(original_image, f"{run_name}_original.png")
    final_path = _save_img(result.final_image, f"{run_name}_final.png") if result.final_image else None

    # ── trace steps ──────────────────────────────────────────────────────────
    trace_records = []
    for ts in result.trace:
        rec: dict = {
            "step": ts.step,
            "think": ts.think,
            "tool": ts.tool_name,
            "params": ts.tool_params,
            "result": ts.result_text,
        }
        if ts.som_image is not None:
            rec["som_image"] = _save_img(
                ts.som_image, f"{run_name}_step{ts.step}_som.png"
            )
        trace_records.append(rec)

    # ── assemble JSON ─────────────────────────────────────────────────────────
    payload = {
        "run_name": run_name,
        "query": query,
        "images": {
            "original": original_path,
            "final": final_path,
        },
        "agent": {
            "answer": result.answer,
            "supporting_mask_ids": result.supporting_mask_ids,
            "n_fp_calls": result.n_fp_calls,
            "n_vlm_calls": result.n_vlm_calls,
            "trace": trace_records,
        },
        "baseline": baseline_answer,
    }

    json_path = out / f"{run_name}.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return str(json_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Grounded visual reasoning with Falcon Perception + local VLM."
    )
    parser.add_argument(
        "--fp-model", default="tiiuae/Falcon-Perception",
        help="Falcon Perception model ID or local path.",
    )
    parser.add_argument(
        "--vlm-model", default="mlx-community/gemma-4-26b-a4b-it-4bit",
        help="Orchestrator VLM model ID or local path.",
    )
    parser.add_argument("--image", required=True, help="Path or URL to input image.")
    parser.add_argument("--query", required=True, help="Natural-language query.")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--fp-max-tokens", type=int, default=512)
    parser.add_argument("--auto-relations-max", type=int, default=6)
    parser.add_argument("--max-corrections", type=int, default=2)
    parser.add_argument("--output", default="grounded_result.png")
    args = parser.parse_args()

    print(f"Loading Falcon Perception: {args.fp_model}")
    fp_model, fp_processor = load(args.fp_model)

    vlm_client = LocalVLMClient(args.vlm_model, max_tokens=args.max_tokens)

    if args.image.startswith("http"):
        import io
        import urllib.request
        with urllib.request.urlopen(args.image) as resp:
            image = Image.open(io.BytesIO(resp.read())).convert("RGB")
    else:
        image = Image.open(args.image).convert("RGB")

    result = run_agent(
        image, args.query, fp_model, fp_processor, vlm_client,
        fp_max_tokens=args.fp_max_tokens,
        auto_relations_max=args.auto_relations_max,
        max_corrections=args.max_corrections,
        verbose=True,
    )

    print(f"\nAnswer: {result.answer}")
    if result.final_image:
        result.final_image.save(args.output)
        print(f"Result image saved to {args.output}")


if __name__ == "__main__":
    main()
