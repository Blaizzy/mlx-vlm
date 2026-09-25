# Models

mlx-vlm began as a vision-first project for running image-understanding models on
Apple silicon, and has since grown to cover the full multimodal spectrum: text-only
LLMs, video, any-to-any omni models, audio (speech in and out), embeddings, and
re-rankers. This catalog groups every supported model **family** by modality; where
a family ships extra documentation, its name links to that model's README. The CLI
(`mlx_vlm.generate` / `mlx_vlm.chat`) and OpenAI-compatible server accept
`--image`, `--audio`, and `--video` inputs depending on the model.

## Vision & Image

Vision-language models (image understanding):

- **Qwen-VL** — Qwen2-VL, Qwen2.5-VL, Qwen3-VL (dense and MoE), and Qwen3.5-VL; a flexible, native-resolution VLM line that also handles video.
- **Gemma 3** — Google's image+text Gemma 3 vision-language models.
- **LLaVA** — the classic LLaVA, LLaVA-NeXT, and LLaVA-Bunny visual-instruction models.
- **Idefics** — Hugging Face Idefics2 / Idefics3 interleaved image-text models.
- **SmolVLM** — compact Idefics3-based VLM for on-device image understanding.
- **Pixtral & Mistral-Small** — Mistral's Pixtral and Mistral-Small-3 vision models.
- **Llama Vision** — Llama 3.2 Vision (mllama) and Llama 4 multimodal.
- **Phi Vision** — Phi-3-Vision and [Phi-4 Reasoning Vision](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/phi4_siglip/README.md).
- **InternVL** — OpenGVLab InternVL chat models.
- **GLM-4V** — Zhipu GLM-4V dense and MoE vision-language models.
- **[MiniCPM-V](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/minicpmv4_6/README.md)** — OpenBMB MiniCPM-V 4.6, a high-resolution multi-image VLM.
- **DeepSeek-VL** — DeepSeek-VL2 and the DeepSeek-VL multi-modality base.
- **Kimi-VL** — Moonshot Kimi-VL, Kimi-K2.5, and Kimi-K3 (MoonViT vision).
- **Molmo** — AllenAI Molmo and Molmo2, plus [MolmoPoint](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/molmo_point/README.md) for visual pointing.
- **[Moondream](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/moondream3/README.md)** — small, fast Moondream2 / Moondream3 VLMs.
- **Florence-2** — Microsoft's encoder-decoder model for captioning, detection, and grounding.
- **PaliGemma** — Google PaliGemma image-text model (single-image).
- **Aya Vision** — Cohere Aya Vision multilingual VLM.
- **Cohere Compass** — Cohere Compass document-understanding vision-language model.
- **[Granite Vision](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/granite4_vision/README.md)** — IBM Granite Vision 3.2 and Granite 4.0 Vision.
- **[ERNIE-VL](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/ernie4_5_moe_vl/README.md)** — Baidu ERNIE 4.5 VL (MoE) vision-language model.
- **Hunyuan-VL** — Tencent Hunyuan vision-language model.
- **FastVLM** — Apple FastVLM (LLaVA-Qwen2 style) for efficient inference.
- **LFM2-VL** — Liquid AI LFM2-VL vision-language model.
- **[PLaMo-VL](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/plamo2vl/README.md)** — Preferred Networks PLaMo 2 VL.
- **Jina-VLM** — Jina AI vision-language model.
- **LLM-jp VL** — Japanese LLM-jp vision-language model.
- **[MiniMax-M3 VL](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/minimax_m3_vl/README.md)** — MiniMax-M3 vision-language model with video support.
- **Muse Glimmer** — Muse Glimmer vision-language model.
- **Mage-VL** — the Qwen3-VL-based conditioner used by Mage-Flow image generation.
- **Youtu-VL** — Tencent Youtu vision-language model.
- **Zaya-VL** — Zaya1 vision-language model.
- **Step-3 VL** — StepFun Step-3 vision-language model (perception encoder).

OCR & document parsing:

- **[DeepSeek-OCR](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/deepseekocr/README.md)** — DeepSeek-OCR and [DeepSeek-OCR-2](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/deepseekocr_2/README.md) high-compression document OCR.
- **[DOTS-OCR](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/dots_ocr/README.md)** — rednote DOTS-OCR / DOTS-MOCR layout-aware document parsing.
- **[GLM-OCR](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/glm_ocr/README.md)** — Zhipu GLM-based OCR model.
- **[PaddleOCR-VL](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/paddleocr_vl/README.md)** — Baidu PaddleOCR-VL document recognition.
- **[Falcon-OCR](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/falcon_ocr/README.md)** — TII Falcon OCR model.
- **[GOT-OCR 2.0](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/got/README.md)** — StepFun GOT-OCR 2.0 unified OCR (documents, tables, formulas, charts).
- **[Unlimited-OCR](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/unlimited_ocr/README.md)** — long-form / high-resolution OCR model.
- **[Nemotron-Parse](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/nemotron_parse/README.md)** — NVIDIA encoder-decoder document parser emitting markdown with boxes.

Detection, segmentation & grounding:

- **[SAM 3](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/sam3/README.md)** — Segment Anything 3 promptable image/video segmentation ([SAM 3.1](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/sam3_1/README.md)).
- **[SAM 3D Body](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/sam3d_body/README.md)** — 3D human body reconstruction from images.
- **[RF-DETR](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/rfdetr/README.md)** — Roboflow RF-DETR real-time object detection.
- **[RT-DETR v2](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/rt_detr_v2/README.md)** — real-time detection transformer.
- **[LocateAnything](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/locateanything/README.md)** — open-vocabulary visual grounding / localization.
- **[Falcon-Perception](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/falcon_perception/README.md)** — TII early-fusion VLM for detection and segmentation from text queries.

Image generation & editing:

- **[FLUX.2](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/flux2/README.md)** — FLUX.2 Klein text-to-image and image-editing diffusion models.
- **[Z-Image](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/z_image/README.md)** — Tongyi-MAI Z-Image text-to-image (Qwen3 text encoder + DiT).
- **[Ideogram 4](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/ideogram4/README.md)** — Ideogram 4 text-to-image with structured caption control.
- **[Mage-Flow](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/mage_flow/README.md)** — Microsoft Mage-Flow native-resolution image generation and editing.
- **[ERNIE-Image](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/ernie_image/README.md)** — Baidu ERNIE-Image text-to-image diffusion.
- **[Prism Bonsai](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/bonsai/README.md)** — Prism Bonsai image-generation model.

## Text

Text-only language models (vendored / ported LLMs with no vision tower):

- **Llama** — Meta Llama and Llama 4 text backbones.
- **Qwen** — Qwen, Qwen2, Qwen3 (dense and MoE) and Qwen3-Next.
- **Gemma** — Gemma, Gemma 2, Gemma 3 text, Gemma 4 text, and RecurrentGemma.
- **Mistral** — Mistral 4, Ministral 3, and Mixtral.
- **Phi** — Phi, Phi-3 (incl. small), Phi-MoE, and Phixtral.
- **DeepSeek** — DeepSeek, DeepSeek-V2/V3/V3.2, and DeepSeek-V4.
- **GLM** — Zhipu GLM, GLM-4, GLM-4 MoE (incl. Lite), and GLM MoE-DSA.
- **Cohere / Command** — Cohere, Cohere2, and Cohere2 MoE.
- **ERNIE** — Baidu ERNIE 4.5 dense and MoE text models.
- **EXAONE** — LG EXAONE, EXAONE 4, and EXAONE MoE.
- **OLMo** — AllenAI OLMo, OLMo2, OLMo3, and OLMoE.
- **Granite** — IBM Granite, GraniteMoE, and Granite MoE Hybrid.
- **Nemotron** — NVIDIA Nemotron, Nemotron-H, and Nemotron-NAS.
- **Hunyuan** — Tencent Hunyuan and Hunyuan V1 dense.
- **MiniCPM** — OpenBMB MiniCPM and MiniCPM3.
- **MiniMax** — MiniMax and MiniMax-M3 text models.
- **Ling / Bailing** — Ant Group Bailing MoE and Bailing MoE Linear.
- **Kimi** — Moonshot Kimi-Linear text model.
- **InternLM** — InternLM2 and InternLM3.
- **GPT-style** — GPT-2, GPT-BigCode, GPT-NeoX, GPT-OSS, StarCoder2, and NanoChat.
- **SSM & hybrids** — Mamba, Mamba2, RWKV7, Jamba, Falcon-H1, PLaMo / PLaMo 2, and HRM-Text.
- **StableLM** — Stability StableLM.
- **OpenELM** — Apple OpenELM.
- **SmolLM3** — Hugging Face SmolLM3.
- **Arcee AFM** — Arcee Foundation Models (AFM dense and AFMoE).
- **Apertus** — Swiss AI Apertus open LLM.
- **Baichuan-M1** — Baichuan-M1.
- **DBRX** — Databricks DBRX MoE.
- **Helium** — Kyutai Helium.
- **dots.llm1** — rednote dots.llm1 MoE.
- **Klear** — Kuaishou Klear.
- **Mellum** — JetBrains Mellum code model.
- **MiMo** — Xiaomi MiMo and MiMo v2 Flash.
- **LoopCoder** — iQuest LoopCoder code model.
- **Lille-130M** — Lille 130M small LM.
- **LongCat-Flash** — Meituan LongCat-Flash (incl. n-gram variant).
- **Seed-OSS** — ByteDance Seed-OSS.
- **SOLAR** — Upstage SOLAR.
- **Step-3.5** — StepFun Step-3.5 text model.
- **TeleChat3** — China Telecom TeleChat3.
- **Youtu-LLM** — Tencent Youtu text model.
- **BitNet** — 1-bit BitNet.
- **Laguna** — Laguna text model.
- **Diffusion LMs** — [Nemotron Labs Diffusion](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/nemotron_labs_diffusion/README.md), [DiffusionGemma](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/diffusion_gemma/README.md), and LLaDA2 MoE block/masked-diffusion text generators.

## Video

Models that accept video input (or generate video):

- **Qwen-VL** — Qwen2-VL through Qwen3-VL and Qwen3.5-VL sample video frames natively.
- **[Gemma 4](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma4/README.md)** — Gemma 4 (and DiffusionGemma) handle video alongside image and audio.
- **[MiniCPM-V 4.6](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/minicpmv4_6/README.md)** — video understanding over sampled frames.
- **[MiniMax-M3 VL](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/minimax_m3_vl/README.md)** — video-capable vision-language model.
- **GLM-4V** — video understanding in the GLM-4V family.
- **InternVL** — video frame understanding in InternVL chat.
- **Qwen3-Omni** — video as part of its omni input (see Omni).
- **SAM 3** — promptable video segmentation and tracking.
- **[MiniMax-H3](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/minimax_h3/README.md)** — text/image/audio-to-video generation with synchronized audio.

## Omni

Any-to-any / multimodal models that handle audio and vision (and video) together:

- **Qwen3-Omni** — Qwen3-Omni MoE, unifying text, image, audio, and video.
- **[Gemma 4](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma4/README.md)** — Gemma 4 / Gemma 4 unified, with image, audio, speech, and video.
- **Gemma 3n** — natively multimodal Gemma 3n with image and audio input.
- **[Phi-4 Multimodal](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/phi4mm/README.md)** — Microsoft Phi-4-Multimodal (vision + speech).
- **[MiniCPM-o](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/minicpmo/README.md)** — OpenBMB MiniCPM-o omni model (vision + audio + speech).
- **Nemotron Nano Omni** — NVIDIA Nemotron-H Nano Omni (vision + speech + video).
- **Inkling** — Inkling omni model spanning vision, audio, speech, and video.

## Audio

Audio-in and/or audio-out (speech) models:

- **[NemotronLabs VoiceChat](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/nemotron_voicechat/README.md)** — NVIDIA full-duplex speech model that listens, transcribes, replies, and synthesizes speech; served over the `/v1/realtime` WebSocket endpoint.

The omni models above (Qwen3-Omni, Phi-4 Multimodal, MiniCPM-o, Gemma 4 / 3n, Nemotron Nano Omni, Inkling) also accept speech input and, in several cases, generate speech.

## Embedding

Embedding models (text and multimodal), served via `/v1/embeddings`:

- **Qwen3 Embedding** — Qwen3-based text embeddings (last-token pooling).
- **Gemma 3 Embedding** — Gemma-3-based text embeddings (EmbeddingGemma).
- **LFM2 Embedding** — Liquid AI LFM2 text embeddings.
- **Ministral 3 Embedding** — Ministral-3-based text embeddings.
- **Llama (bidirectional)** — bidirectional Llama encoder for text embeddings.
- **SigLIP** — CLIP-style joint image-text embeddings.
- **ColPali-style** — ColQwen2.5 and ColIdefics3 multi-vector visual-document retrieval.
- **Qwen3-VL Embedding** — multimodal embeddings from Qwen3-VL.
- **Llama Nemotron VL Embedding** — multimodal embeddings from Llama Nemotron VL.

The BERT-family encoders listed under Re-rankers (BERT, ModernBERT, XLM-RoBERTa) also serve as text-embedding backbones.

## Re-rankers

Reranking models, served via the server's reranking endpoint:

- **BERT** — cross-encoder (sequence-classifier) reranker.
- **ModernBERT** — modern cross-encoder reranker.
- **XLM-RoBERTa** — multilingual cross-encoder reranker (e.g. BGE-reranker-style).
- **Qwen3** — generative text reranker.
- **Qwen3-VL** — generative vision-language reranker.
