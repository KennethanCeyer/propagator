# Propagator multimodal training ceiling audit

Date: 2026-06-20 UTC

Scope: current repository, current `outputs/propagator-multimodal` run artifacts, training log `logs/train_20260619T121811Z.log`, cache metadata, dataset mix files, model/training code in `train.py`, smoke checks in `scripts/smoke_training.py`, and public SL2610-class device information.

This audit treats Propagator as a stateful recurrent matrix-memory multimodal model, not as a token-indexed KV-cache Transformer.

Implementation status:

- Added `data/regression/sample_05_format_following.jsonl`.
- Added `data/propagator_instruction_balanced_seed.jsonl`.
- Added `data/propagator_dataset_mix_balanced_v2.json`.
- Added `scripts/audit_prop_regressions.py` for deterministic protocol/mask checks and lightweight data/cache imbalance reporting.
- Added recognition-only `[IMAGE_IN]` as an explicit special protocol token in `train.py`.
- The script uses the real `train.py` protocol builder through the repository virtualenv and does not run training.
- Changed `scripts/train.sh` so future launches default to `VALIDATION_CONTROL_BATCHES=8` instead of `0`.
- Changed `scripts/train.sh` so regular training defaults to `data/propagator_dataset_mix_balanced_v2.json` instead of the old imbalanced mix, unless `DATASET_MIX_FILE` is explicitly overridden.
- Changed early stopping in `train.py` from weighted validation CE to a Propagator-specific composite validation score.

Local verification:

```bash
./.venv/bin/python scripts/audit_prop_regressions.py --protocol-only
./.venv/bin/python scripts/audit_prop_regressions.py
./.venv/bin/python -m py_compile scripts/audit_prop_regressions.py
python -m json.tool data/propagator_dataset_mix_balanced_v2.json >/dev/null
bash -n scripts/train.sh
```

Observed result:

- `sample_05_format_following`: 17-token stream, `[USER_END]` at index 10, `[MODEL]` at 11, `[TEXT_OUT]` at 12, `[MODEL_END]` at 14, decoded target `yes`.
- `sample_05_format_following_chunked`: 90-token stream, response phase starts at index 84 with unroll length 32, decoded target `yes`.
- Synthetic audio alignment check: 2 output audio frames, `[MODEL] -> [AUDIO_OUT]` at weight 2.0, q0-q7 frame labels preserved, `[AUDIO_END]` at weight 2.0, then `[MODEL_END]`.
- Image protocol check: `[IMAGE_IN]` appears once in the user context, targets `[LISTEN]` with weight 0.05, never appears as an output target, and the answer path selects `[TEXT_OUT]` then `[MODEL_END]`.
- Runtime recurrent-state check: tiny 1-layer Propagator initializes memory as shape `(2, 8, 8)`, reset mask changes lane sums to `[0.0, 64.0]`, `valid=False` lane memory norm remains `0.0`, and carried-vs-reset path delta is `0.020953`.
- Recurrent source invariant check: `StatefulChunkSampler`, `reset_mask`, `stop_gradient`, `forward_with_memories`, and `chunk_positions` are present. Reset/carry is controlled by reset masks and stream state.
- Current generated `step_300000/sample_05_format_following.txt`: selected `[TEXT_OUT]`, did not terminate with `[MODEL_END]`, emitted 255 content tokens, and started with `Propagator.1.1)...` instead of `yes`.
- Current run config audit: `stateful_train=true`, `stateful_validation=true`, `train_unroll_len=32`, `validation_control_batches=0`, audio backend `mimi`, sample rate 24000, codebooks 8, codebook size 2048.
- Current validation metrics audit: `text_task_acc=0.3533`, `asr_task_acc=0.4599`, `tts_task_acc=0.3709`, `duplex_task_acc=NaN`, `audio_all_codebook_frame_exact_acc=0.0`.
- Current validation composite audit: `validation_composite_score=0.4819`, `validation_protocol_score=0.8203`, `validation_text_score=0.3588`, `validation_speech_score=0.3965`, `validation_audio_aux_score=0.0782`, `validation_duplex_score=0.0`, `validation_coverage_score=1.0`.
- Current data/cache audit warnings: posttrain contains 5.0% name-question rows and 5.0% repeated code-word rows; train cache is 74.9% plain text; validation cache is 94.8% plain text.
- Balanced v2 mix audit: total configured weight 1.0, 38.0% duplex chat, 8.0% Dolly instruction, 9.0% plain text, 36.0% audio ASR/TTS sources, 9.0% EchoX hybrid. Identity configured share is 1.0%; `sample_05_format_following` fixture share is 9.0%.
- Balanced v2 local-tokenize dry-run: `propagator_instruction_balanced_seed.jsonl`, `sample_05_format_following.jsonl`, and `propagator_identity.jsonl` all pass `tokenize_duplex` on checked rows. The old `propagator_posttrain_10k.jsonl` is excluded from balanced v2 because it is repetitive and contains raw modality markers in assistant text.
- `scripts/train.sh` passes syntax check.
- Regular `scripts/train.sh` launches now default to the balanced v2 mix; posttrain mode can still use its own posttrain mix.
- Early stopping now maximizes `validation_composite_score`; `val_loss` is still recorded and plotted, but it is not the sole stop criterion.

## 0. Current run status

The current run did not hang. It exited normally at step 300000 due to early stopping.

Evidence:

| Item | Observation |
| --- | --- |
| Planned schedule | `steps_per_epoch=38604`, `epoch_total_steps=1158120`, `max_steps=0`, `total_steps=1158120` |
| Actual stop | `[Early Stop] stopping at step 300000: validation did not improve by 0.01 for 12 evaluations` |
| Lock | training lock released |
| PID | recorded process no longer running |
| Latest checkpoint/output | `outputs/propagator-multimodal/step_300000` |

This matters because the model did not reach the intended 1.158M-step schedule. More importantly, early stopping was driven by a validation mix that does not adequately measure the failure modes the user cares about.

## 1. Executive diagnosis

The most likely bottleneck is not numerical instability. The model is optimizing and has learned easy protocol and identity transitions. The ceiling is caused by a combination of data mixture, curriculum, validation, and missing architecture-specific diagnostics.

Highest-confidence issues:

| Priority | Diagnosis | Evidence |
| --- | --- | --- |
| P0 | Validation and early stopping are not aligned with required behavior. | Validation uses only 16 batches and `validation_control_batches=0`; format following, memory recall, hybrid dialogue, and image grounding are not part of the stop criterion. The run stopped at 300k before the intended 1.158M steps. |
| P0 | Active dataset mix is dominated by plain text continuation. | Train cache: FineWeb 56.66%, Wikipedia 18.21%. Validation cache: FineWeb 71.72%, Wikipedia 23.05%. This trains continuation more than instruction-following memory. |
| P0 | Strict format following is underrepresented and repetitive. | `data/propagator_posttrain_10k.jsonl` contains 500 `What is your name?` rows and 500 repeated "Repeat only the code word" rows. It does not provide enough diverse constrained-output supervision. |
| P0 | `sample_05_format_following` shows memory/semantic retrieval failure, not only a protocol failure. | The model emits `[TEXT_OUT]` correctly, then generates identity and unrelated repetitive text instead of one word. |
| P1 | Matrix-memory learning is not directly measured. | No logged diagnostics for memory norm, eta/forget distribution, read/write cosine, reset counts, state leakage, or chunk-boundary recall. |
| P1 | State reset depends on external sampler reset masks, not the `[SESSION]` token itself. | Code resets recurrent memory through `reset_mask`; `[SESSION]` is a learned protocol token but not an intrinsic hard reset. This is acceptable only if every training/eval/runtime path reliably resets state. |
| P1 | Truncated BPTT unroll length is likely too short for delayed constraints. | `train_unroll_len=32`; gradients stop at chunk boundaries. Long prompts and format constraints can cross chunks, but the supervised credit path is only one chunk. |
| P1 | Audio evaluation is too sparse and the metric suite is misleading. | TTS task accuracy is ~0.37, ASR ~0.46, codebook exact-frame accuracy ~0, aux token accuracy ~0.078. `duplex_task_acc` is NaN and active run logs `audio_to_audio_samples: 0`. |
| P1 | Active run did not train/evaluate real image recognition. | No `[IMAGE_IN]` protocol path or vision-token stream exists in the current audited implementation. |
| P2 | MoE is risky for edge deployment. | Current model uses `moe_experts=2`, `top_k=1`; sparse routing can complicate NPU/compiler support and deterministic batch-1 latency. |

The model currently appears to learn shallow protocol transitions and some local text/audio token statistics, but it does not yet demonstrate reliable matrix-memory storage and retrieval of instruction constraints.

## 2. Architecture-specific audit

### 2.1 Matrix memory implementation

Current block behavior:

| Component | Current behavior | Audit result |
| --- | --- | --- |
| Persistent matrix memory | Per-layer memory shape follows fixed key/value matrix dimensions. | Correct architectural direction for Propagator. Needs instrumentation. |
| Read key | Projected from normalized residual stream, grouped into associative lanes, RMS-normalized, optionally stream-position rotated. | Plausible. Need read-key entropy/collapse diagnostics. |
| Write key | Separate projected write key, grouped and normalized. | Plausible. Need write-key lane utilization diagnostics. |
| Value projection | `tanh(write_value_proj(w))`. | Stable but may saturate. Need saturation histogram. |
| Delta-rule update | Uses error `write_value - value_hat`, clipped to [-1, 1]. | Plausible associative delta rule. Clipping avoids explosion but may hide overwrite/interference. |
| Eta/update rate | `sigmoid(write_gate) * write_rate`, default `write_rate=0.02`. | Conservative. May underwrite long constraints when prompts are noisy. |
| Forget gate | `sigmoid(forget_gate) * forget_rate`, default `forget_rate=0.002`. | Conservative forgetting. Good for stability, but stale interference risk is unmeasured. |
| Grouped key lanes | `associative_groups=4`. | Reasonable, but no per-group utilization metrics. |
| RoPE-style stream rotation | Positions are `chunk_position * train_unroll_len + local_position`. | Good for continuity if stream positions are correct. Needs test that reset clears position/memory together. |
| Residual stability | RMSNorm + small residual gammas initialized near 0.1. | Stable, likely why training is numerically fine. |
| SwiGLU MLP | Enabled. | Good default. |
| Optional MoE | Enabled with 2 experts, top-1. | Research okay; edge questionable. |
| Final heads | Text/control LM head plus audio auxiliary codebook heads. | Architecturally coherent, but evaluation and loss weighting need redesign. |

### 2.2 Stateful processing and BPTT

Current behavior:

- `stateful_train=true` and `stateful_validation=true`.
- Recurrent memories are carried across chunks for the same stream.
- `forward_with_memories` returns `stop_gradient(final_memories)`.
- Effective gradient-through-time is the current chunk only, currently 32 tokens.
- Chunk-to-chunk state is persistent in activations, but training credit assignment does not cross chunk boundaries.

Interpretation:

This is a correct truncated-BPTT design, but the current curriculum does not force the model to learn delayed constraint storage. If a format instruction appears in one chunk and the answer begins in a later chunk, the model can only learn from short local gradients unless many examples repeatedly teach the same memory update/retrieval pattern.

### 2.3 Reset and isolation risk

Important finding:

`[SESSION]` is not a hard-coded memory reset. The memory reset is controlled by sampler/runtime reset masks. This is acceptable only if every path is disciplined. It is also a silent bug class because a valid-looking token stream can still leak memory if the external reset is wrong.

Required tests:

- A two-sample batch where sample B starts with `[SESSION]` but reset is intentionally disabled should fail the state-leak detector.
- The normal sampler path should prove that unrelated samples receive reset masks and stream position reset together.
- Runtime generation should prove zero initial memory plus `[SESSION]` matches training-time first-session behavior.

### 2.4 Can the model store instruction constraints?

Current evidence says no, not reliably.

`sample_05_format_following` has a simple user constraint: "Answer with one word: is water wet?" The model selected text output mode, but generated identity and unrelated text. This means:

- Protocol transition `[MODEL] -> [TEXT_OUT]` is learned.
- The one-word constraint was not retrieved at response time.
- Semantic answer selection was not reliable.
- Identity/name data is intruding into unrelated prompts.

This is not a KV-cache problem. It is a matrix-memory write/read/curriculum problem: the model must learn to write the constraint during the user phase and retrieve it during the model phase.

## 3. Protocol-token and stream audit

### 3.1 Current protocol format

The audited builders use the expected high-level sequence:

```text
[SESSION]
[USER]
user content...
[USER_END]
[MODEL]
[TEXT_OUT] or [AUDIO_OUT] or [HYBRID_OUT]
model content...
[MODEL_END]
```

During user input, targets are generally `[LISTEN]` until the final user token, whose target is `[USER_END]`. Then `[USER_END]` targets `[MODEL]`. Then `[MODEL]` targets the output modality token.

### 3.2 Loss weighting

Current notable weights and caveat:

| Target type | Weight |
| --- | --- |
| `[LISTEN]` | 0.05 |
| `[TEXT_OUT]`, `[AUDIO_OUT]`, `[HYBRID_OUT]`, `[AUDIO_END]` via `default_loss_weight_for_target` | 2.0 |
| `[MODEL] -> [TEXT_OUT]` in the traced `tokenize_duplex` path | `content_loss_weight`, currently 1.0 |
| Other control tokens | 1.0 |
| Text/audio content | usually 1.0 |

This means the intended modality-token emphasis is not uniformly applied. In the traced `sample_05_format_following` fixture, `[MODEL] -> [TEXT_OUT]` is supervised, but only at weight 1.0. This is not the root cause of the sample failure, but it is a code path to inspect before relying on modality weights.

### 3.3 Training vs inference

The generation path mostly mirrors the training protocol, but has important differences:

- User-side decisions are coerced toward `[LISTEN]` unless `[USER_END]` is selected.
- Output mode after `[MODEL]` is sampled rather than forced in the normal sample path.
- Sample generation uses temperature/top-k, which is fine for qualitative samples but poor for a regression test.

Recommendation:

Keep generative samples, but add deterministic protocol regression runs that:

- inspect raw logits for `[LISTEN]`, `[USER_END]`, `[MODEL]`, `[TEXT_OUT]`, and `[MODEL_END]`;
- force the runtime protocol only where the product runtime will force it;
- separately score unforced protocol accuracy.

## 4. Debug: `sample_05_format_following`

### 4.1 Observed failure

At step 300000, the prompt is effectively:

```text
Answer with one word:
is water wet?
```

The model output begins:

```text
[TEXT_OUT] -> P -> rop -> ag -> ator -> . -> 1 -> . -> 1 -> ...
```

This is a severe failure. It is not just "not exact enough." It violates:

- one-word constraint;
- semantic answer;
- no identity intrusion;
- no repetition;
- correct answer termination.

### 4.2 Most likely causes

| Cause | Likelihood | Reason |
| --- | --- | --- |
| Insufficient format-following diversity | Very high | Current posttrain data contains repeated code-word and identity rows, not broad schemas and one-word constraints. |
| Plain-text continuation dominance | Very high | Active train cache is 74.87% FineWeb/Wikipedia; validation is 94.77% FineWeb/Wikipedia. |
| Early stopping metric mismatch | Very high | No dedicated format or memory recall validation was part of early stop. |
| Identity intrusion from over-repeated identity/name rows | High | 500 name-question rows in posttrain plus identity data repeated in training. |
| Truncated BPTT too short for delayed constraints | Medium-high | Unroll length is 32. The sample itself is short, but the general mechanism for constraint storage is not trained. |
| Protocol construction bug for this sample | Medium | Needs a fixture-level token/mask test. The visible output mode was correct, so protocol is not the only issue. |
| Wrong state reset at generation | Medium | Current generation starts from zero memory, but reset behavior must be tested against training streams. |
| Audio/codebook bug | Low for this sample | This is a text-only format failure. |

### 4.3 Permanent regression requirements

Create a deterministic regression fixture named `sample_05_format_following` with at least these checks:

| Check | Requirement |
| --- | --- |
| Tokenized user phase | The format instruction appears before `[USER_END]`. |
| Target stream | The supervised answer is exactly a one-word answer, followed by `[MODEL_END]`. |
| Loss mask | User prompt tokens are not supervised as assistant content; answer tokens are supervised. |
| Protocol | User input targets `[LISTEN]`, final user token targets `[USER_END]`, `[USER_END]` targets `[MODEL]`, `[MODEL]` targets `[TEXT_OUT]`. |
| Chunking | A variant splits instruction and answer across chunks; state must carry the constraint. |
| Deterministic decode | Greedy output must be exactly one word and then `[MODEL_END]`. |
| Intermediate behavior | Raw logits should rank `[TEXT_OUT]` correctly after `[MODEL]` and rank `[MODEL_END]` after the one-word answer. |

## 5. Audio codec and TTS audit

### 5.1 Active codec path

The active run uses Mimi, not EnCodec.

Observed active config:

| Setting | Value |
| --- | --- |
| Backend | `mimi` |
| Sample rate | 24000 Hz |
| Frame rate | 12.5 fps |
| Codebooks | 8 |
| Codebook size | 2048 |
| Auxiliary codebooks | q1-q7 via audio aux heads |

The README mentions multiple codec options, but current run artifacts clearly identify Mimi.

### 5.2 Audio supervision structure

Current design:

- Main LM vocabulary predicts the primary audio codebook token.
- Auxiliary codebook heads predict remaining codebooks.
- Pretokenized Mimi datasets are loaded as ASR or TTS task forms.
- Audio code frames are packed into per-frame codebook labels.

This structure is reasonable, but the current training/evaluation does not prove that paired speech alignment is working.

### 5.3 Current metrics

At step 300000:

| Metric | Value |
| --- | --- |
| `asr_task_acc` | 0.4599 |
| `tts_task_acc` | 0.3709 |
| `audio_token_acc` / `audio_main_acc` | 0.3586 |
| `audio_aux_token_acc` | 0.0782 |
| `audio_codebook_acc` | 0.0003686 |
| `audio_all_codebook_frame_exact_acc` | 0.0 |
| `duplex_task_acc` | NaN |

Interpretation:

- Exact all-codebook frame accuracy is too harsh as the primary TTS metric.
- Auxiliary token accuracy is far above random for 2048-way codebooks, so the model is learning some audio distribution.
- TTS quality plateau around 0.35-0.4 is likely caused by insufficient paired TTS/hybrid coverage, high codec-token entropy, task mixture imbalance, and inadequate audio-specific curriculum.
- `duplex_task_acc=NaN` and `audio_to_audio_samples: 0` show that the run is not exercising the full requested speech-dialogue behavior.

### 5.4 Audio fixes

Required:

- Log per-codebook accuracy q0-q7.
- Log first-codebook vs delayed-codebook accuracy if a delay pattern is used.
- Decode fixed TTS probes and measure ASR intelligibility with a separate ASR model.
- Track duration error and early/late `[AUDIO_END]`.
- Separate ASR, TTS, audio-to-audio, and hybrid validation streams.
- Ensure paired text/audio examples survive filtering and chunk packing as one stream.
- Use a speech curriculum before fully mixed multimodal training.

## 6. Dataset mixture redesign

### 6.1 Current active mixture problem

Active cache composition:

| Split | FineWeb | Wikipedia | Instruction/identity local rows | Audio ASR/TTS |
| --- | ---: | ---: | ---: | ---: |
| Train chunks | 56.66% | 18.21% | 1.24% | about 23.9% |
| Validation chunks | 71.72% | 23.05% | 0.16% | about 5.1% |

The validation split is especially damaging: it tells early stopping that plain text continuation is the main task.

### 6.2 Repetitive sample problem

`data/propagator_posttrain_10k.jsonl` contains:

| Pattern | Count |
| --- | ---: |
| `What is your name?` | 500 |
| `Repeat only the code word...` | 500 |

This explains identity intrusion and weak generalization. The current data teaches a narrow set of shallow patterns rather than broad instruction following.

### 6.3 Recommended final mixed-training sampler weights

These are sampler probabilities after filtering/capping, not raw dataset sizes.

| Bucket | Weight | Cap policy | Notes |
| --- | ---: | --- | --- |
| Broad text instruction SFT | 18% | per-source and per-template cap | General assistant behavior. |
| Strict format/schema following | 12% | high oversampling, dedup by schema/task | JSON, YAML, Markdown table, bullet list, one-word, regex/schema constraints. |
| Extraction/classification | 8% | balanced labels | Entity extraction, sentiment, routing, binary/multiclass. |
| Summarization/paraphrase | 7% | source-balanced | Short and medium contexts. |
| Short reasoning/command following | 8% | template-balanced | Arithmetic, ordering, device commands, safety boundaries. |
| Matrix-memory recall curriculum | 10% | generated probes plus heldout variants | Delayed constraints, chunk splits, contradiction updates, multi-turn recall. |
| Multi-turn dialogue/interruption | 7% | cap identity/name to tiny fraction | Includes `[USER_INTERRUPT]` and state update. |
| Plain text continuation | 8% | strict maximum | Keep language fluency but stop it dominating instruction behavior. |
| ASR audio-to-text | 8% | speaker/source balanced | Audio input, text output. |
| TTS text-to-audio | 8% | speaker/source balanced | Text input, audio output. |
| Audio dialogue/hybrid/audio-to-audio | 4% | oversample rare tasks | Audio-to-audio, audio-to-hybrid, text+audio tasks. |
| Image recognition/QA | 7% | object/scene/source balanced | Recognition-only image context. |
| Identity/self-description | 1% | tiny, diverse, no repeats | Enough for name, not enough to intrude. |
| Refusal/boundary | 2% | policy/task balanced | Only if product requires assistant safety boundaries. |

Total: 100%.

### 6.4 Caps and logging requirements

Do not use one global cap. Implement and log:

- per-source cap;
- per-task cap;
- per-modality cap;
- effective examples after filtering;
- effective text tokens after packing;
- effective audio frames after packing;
- effective image examples after packing;
- per-bucket validation sample counts;
- sampler lane allocation by bucket;
- per-bucket loss and accuracy.

Use temperature sampling or explicit weighted lane assignment so a huge text dataset cannot silently erase rare but important tasks.

## 7. Text/audio pairing strategy

Paired examples must remain paired through preprocessing and packing. Required task forms:

| Task form | Input protocol | Output protocol | Supervision requirement |
| --- | --- | --- | --- |
| Text to text | `[TEXT_IN] text` | `[TEXT_OUT] text` | Target only assistant side plus control transitions. |
| Audio to text | `[AUDIO_IN] audio` | `[TEXT_OUT] text` | Audio input not trained as text answer; text answer supervised. |
| Text to audio | `[TEXT_IN] text` | `[AUDIO_OUT] audio` | Audio codebooks supervised only on output side. |
| Audio to audio | `[AUDIO_IN] audio` | `[AUDIO_OUT] audio` | Directionality explicit; no ASR/TTS confusion. |
| Audio to hybrid | `[AUDIO_IN] audio` | `[HYBRID_OUT] text + audio` | Both text and audio labels aligned. |
| Text+audio to text | `[TEXT_IN] text [AUDIO_IN] audio` | `[TEXT_OUT] text` | Context modalities both in user phase. |
| Text+audio to audio | `[TEXT_IN] text [AUDIO_IN] audio` | `[AUDIO_OUT] audio` | Prompt side unsupervised as output. |

Add a preprocessing audit that samples packed rows and verifies:

- modality token placement;
- boundary token placement;
- target mask;
- audio lane labels;
- codebook shift;
- chunk boundaries;
- stream ID continuity;
- reset only at true session start.

## 8. Image recognition integration

### 8.1 SL2610-class deployment constraints

Public sources indicate the target class is Synaptics Astra SL2610/SL2619:

- Synaptics describes SL2610 as an Arm Cortex-A55/Cortex-M52/Mali GPU edge AI line with Torq and Coral NPU support, and compiler/runtime support based on IREE/MLIR for LiteRT, PyTorch, ONNX, and JAX.
- Google Coralboard documentation lists a Synaptics Astra SL2619 SoC with Coral NPU, 1 TOPS Torq inference engine, 2GB DDR4 with optional 1GB, camera/display support over CSI/DSI and USB, and Yocto Linux.
- TechNexion OSM-SL2610 product brief lists Cortex-A55 up to 2GHz, Google Kelvin Core AI/NPU 1 TOPS, and up to 2GB LPDDR4.
- Synaptics SL2610 datasheet Rev D describes the Torq T1 NPU as up to 1 TOPS with 512 parallel compute elements, int8/int16/BFLOAT16 support, arbitrary layer execution through an integrated Coral NPU RISC-V core, and MLIR/IREE targeting.

Source URLs:

- https://www.synaptics.com/products/embedded-processors/sl2610-product-line
- https://developers.google.com/coral/products/SL2610-dev-board
- https://cp.synaptics.com/cognidox/download/NR-160466-DS-APPROVED.pdf
- https://www.technexion.com/products/system-on-modules/osm/osm-sl2610/

Conclusion:

Assume about 2GB system memory and about 1 TOPS NPU-class acceleration. Do not design a large image tower. Do not add image generation. Use small recognition-only visual context.

### 8.2 Recommended protocol

Add explicit image input tokens:

```text
[SESSION]
[USER]
[IMAGE_IN] <visual_tokens>
[TEXT_IN] question
[USER_END]
[MODEL]
[TEXT_OUT] answer
[MODEL_END]
```

Do not add image output tokens.

### 8.3 Image design options

| Option | Recommendation | Details |
| --- | --- | --- |
| A. Frozen tiny vision encoder | Best long-term edge path | 160x160 or 192x192 input first. 224x224 only after measured latency/memory. Project a small number of visual tokens into Propagator hidden size. |
| B. Precomputed visual tokens | Best near-term training path | Train the language/memory side with stored embeddings before committing to an edge encoder. Allows fast data iteration. |
| C. Very small CNN/MobileNet tokenizer | Best fallback for NPU/operator simplicity | Produce 16-64 visual tokens. Favor depthwise convs and simple ops supported by mobile compilers. |

Recommended order:

1. Implement Option B for training and protocol validation.
2. Train image-grounded recognition tasks with frozen projected visual tokens.
3. Select Option A or C after benchmarking Torq/Coral compiler support on the actual SKU.

Repository status:

- `[IMAGE_IN]` is now an explicit special token.
- The current implementation treats it as recognition-only user context, not an output target.
- The regression audit verifies `[IMAGE_IN] -> [LISTEN]` behavior and `[TEXT_OUT]` answer supervision.
- Actual visual embeddings are not implemented yet; current image seed rows are protocol/recognition scaffolding.

SL2610 image-resolution recommendation:

- Start at 160x160 for the edge path.
- Allow 192x192 only after measured batch-1 latency and memory are acceptable.
- Do not use 224x224 by default under the 2GB/1 TOPS constraint.
- Prefer 16-64 projected visual tokens, not dense high-resolution visual sequences.

### 8.4 Image tasks

Add:

- image captioning;
- object recognition;
- simple scene understanding;
- camera-frame QA;
- text+image instruction following;
- hallucination resistance when requested content is absent;
- audio+image+text grounding where feasible.

## 9. Training curriculum

Do not train the full mixture from step 0. Use staged training with stage-specific validation and early stopping.

### Stage A: Protocol and text semantic stabilization

Goal:

- strong text semantics;
- instruction following;
- format following;
- stable protocol transitions.

Mixture:

| Bucket | Weight |
| --- | ---: |
| Broad text instruction | 35% |
| Strict format/schema | 25% |
| Extraction/classification/summarization/paraphrase | 20% |
| Protocol/control synthetic | 8% |
| Matrix recall short tasks | 8% |
| Identity | 2% |
| Refusal/boundary | 2% |

Disable early stopping on plain validation CE alone. Use composite validation including format exactness and protocol accuracy.

### Stage B: Matrix-memory recall curriculum

Goal:

- teach recurrent matrix memory to preserve constraints across chunks.

Tasks:

- format constraints placed early in stream;
- answer requirements after one or more chunk boundaries;
- multi-turn recall;
- contradiction updates;
- interruptions;
- delayed response tasks;
- state reset and leakage tests.

Increase unroll length for part of training if memory permits, or use mixed unroll lengths such as 32/64/128. If keeping 32, oversample chunk-boundary recall heavily.

### Stage C: Speech alignment

Goal:

- learn ASR, TTS, audio-dialogue alignment.

Mixture:

| Bucket | Weight |
| --- | ---: |
| ASR | 25% |
| TTS | 30% |
| Audio-to-audio | 10% |
| Audio-to-hybrid | 10% |
| Text+audio to text/audio | 10% |
| Text instruction refresh | 10% |
| Format/protocol refresh | 5% |

Track ASR WER, TTS intelligibility, duration, audio repetition, per-codebook metrics.

### Stage D: Mixed multimodal training

Goal:

- stable modality switching and grounding.

Mixture:

| Bucket | Weight |
| --- | ---: |
| Text instruction/format/memory | 40% |
| Speech tasks | 35% |
| Image recognition/QA | 20% |
| Protocol/interruption/control | 5% |

### Stage E: Instruction post-training

Goal:

- usable assistant behavior with Propagator protocol.

Include:

- stateful protocol formatting;
- delayed constraints;
- user interruption;
- modality selection;
- text instruction to text;
- text instruction to audio;
- audio instruction to text;
- image+text instruction to text;
- hybrid context examples.

### Stage F: Distillation and edge compression

Goal:

- fit SL2610-class target.

Use:

- larger research teacher;
- compact dense student;
- 4-bit quantization;
- recurrent-state accounting;
- audio buffer accounting;
- image encoder benchmark;
- batch-1 latency measurement on target compiler/runtime.

## 10. Model scale and deployment

### 10.1 Current research model

Current run config:

| Parameter | Value |
| --- | --- |
| Hidden size | 768 |
| Layers | 16 |
| Memory key size | 192 |
| Memory value size | 384 |
| Associative groups | 4 |
| MoE | 2 experts, top-1 |
| Train unroll | 32 |
| Batch size | 128 |
| Codec | Mimi, 8 codebooks, 2048 size |

This scale should be able to learn simple instruction following if the data/curriculum are correct. The fact that it fails `sample_05_format_following` is more likely data/training/eval/memory-curriculum than raw parameter count.

Research recommendation:

- keep or slightly increase current model for teacher training;
- increase validation coverage before increasing size;
- add memory diagnostics before blaming architecture capacity;
- consider longer or mixed unroll training for memory recall stages.

### 10.2 Compact SL2610-class model

Current edge report indicates about 220M parameters and about 110MB at 4-bit, with batch-1 recurrent state around 4.5MB. Parameters alone fit a 2GB target. The real risk is runtime:

- activations;
- KV-free recurrent matrix memory;
- audio codec buffers;
- image encoder buffers;
- compiler-supported ops;
- CPU fallback;
- MoE routing overhead;
- quantization of recurrent updates.

Edge recommendation:

- prefer a dense compact student over MoE unless target compiler proves sparse routing is efficient;
- use 4-bit weights but keep critical recurrent state/update math in fp16/int8-tested form if quality requires;
- use small vision tokenizer;
- keep image resolution 160x160 first, 192x192 if measured safe, 224x224 only if latency/memory headroom is proven;
- benchmark with batch 1 and streaming chunks, not offline batch throughput.

## 11. Propagator-specific instruction tuning

Do not use generic chat SFT formatting. Use the exact runtime protocol builder for training, evaluation, and deployment.

Instruction-tuning data must include:

- `[SESSION]`, `[USER]`, `[LISTEN]`, `[USER_END]`, `[MODEL]`, output modality, `[MODEL_END]`;
- strict format examples;
- delayed constraints across chunks;
- multi-turn state updates;
- interruptions;
- modality selection examples;
- text instruction to text answer;
- text instruction to audio answer;
- audio instruction to text answer;
- image+text instruction to text answer;
- hybrid context examples.

Recommended SFT phases:

| Phase | Purpose |
| --- | --- |
| SFT-1 | Text instruction and format behavior with deterministic protocol. |
| SFT-2 | Delayed memory constraints and chunk-boundary recall. |
| SFT-3 | Audio modality alignment. |
| SFT-4 | Image recognition/context grounding. |
| SFT-5 | Mixed runtime-style conversations. |

## 12. DPO / RLHF alignment plan

Do not begin RLHF now. The supervised model is not yet competent enough.

Correct sequence:

1. Fix preprocessing, masking, protocol tests, dataset mixture, and validation.
2. Train strong SFT behavior.
3. Build preference pairs.
4. Start with DPO or another offline preference method.
5. Consider online RLHF only after basic competence exists.

Preference comparisons:

| Area | Preferred vs rejected |
| --- | --- |
| Format following | exact schema/one-word output vs verbose or invalid output |
| Semantics | correct answer vs plausible nonsense |
| Modality | correct `[TEXT_OUT]`/`[AUDIO_OUT]`/`[HYBRID_OUT]` vs wrong mode |
| TTS | intelligible, correct duration, semantically consistent vs collapsed/repetitive audio |
| Protocol | stable termination vs missing `[MODEL_END]` or user-token emission |
| Image grounding | grounded answer vs hallucinated object |
| Interruption | updates state correctly vs continues stale answer |
| Matrix recall | delayed constraint retained vs forgotten |

Audio preference signals:

- ASR-based intelligibility;
- duration control;
- silence avoidance;
- repetition avoidance;
- audio-token collapse detection;
- text/audio semantic consistency;
- human naturalness labels where possible.

## 13. Evaluation suite

Add architecture-matched evaluation, not only token CE.

Required metrics:

| Category | Metrics |
| --- | --- |
| Protocol | transition accuracy, `[LISTEN]`, `[USER_END]`, `[MODEL]`, `[MODEL_END]`, output modality accuracy |
| Text | semantic correctness, exact format, JSON validity, schema adherence, repetition |
| Matrix memory | recall across chunks, delayed constraints, contradiction update, state reset, state leakage |
| Interruption | interruption detection, stale-state suppression, response restart |
| Audio | ASR WER, TTS intelligibility, duration error, per-codebook accuracy, collapse/repetition, text-audio consistency |
| Hybrid | text+audio response consistency, correct mode selection |
| Image | captioning, object recognition, scene QA, hallucination resistance |
| Edge | 4-bit quality, batch-1 latency, memory peak, CPU fallback count |

Early stopping should use a composite score, for example:

```text
0.25 * text_instruction_score
+ 0.20 * format_exact_score
+ 0.15 * matrix_recall_score
+ 0.15 * protocol_score
+ 0.15 * speech_score
+ 0.10 * multimodal_grounding_score
```

For stages before image integration, redistribute the image component across text/memory/speech.

## 14. Regression-test plan for `sample_05_format_following`

Create a permanent test fixture:

```json
{
  "id": "sample_05_format_following",
  "messages": [
    {"role": "user", "content": "Answer with one word: is water wet?"},
    {"role": "assistant", "content": "yes"}
  ]
}
```

Required variants:

| Variant | Purpose |
| --- | --- |
| Short one-chunk | Basic exact output. |
| Instruction split before answer chunk | Proves matrix-memory carry. |
| Distractor identity prompt earlier in same batch | Detects identity intrusion. |
| Separate previous sample in same lane | Detects reset leakage. |
| Contradictory later constraint | Proves update/overwrite behavior. |

Assertions:

- tokenized input includes `[TEXT_IN]` or equivalent text-mode marker in user phase;
- user target sequence is `[LISTEN]` until `[USER_END]`;
- `[USER_END] -> [MODEL]`;
- `[MODEL] -> [TEXT_OUT]`;
- target answer is exactly `yes`;
- target after `yes` is `[MODEL_END]`;
- loss mask covers assistant answer and control transitions;
- greedy decode emits exactly one content word;
- decode terminates with `[MODEL_END]`;
- no `[USER]`, `[LISTEN]`, `[USER_END]`, or identity phrase appears in model content.

Implemented preprocessing-level regression:

```bash
./.venv/bin/python scripts/audit_prop_regressions.py --protocol-only
```

This verifies the real `train.py` protocol builder for the short and chunk-boundary `sample_05_format_following` fixtures. It checks the user-side `[LISTEN]` targets, `[USER_END] -> [MODEL]`, `[MODEL] -> [TEXT_OUT]`, exact one-word target text, answer supervision, `[MODEL_END]`, and that the chunked variant reaches the response phase after the configured unroll boundary.

## 15. Suspected code bugs or silent failure points to inspect first

Prioritized:

1. Validation set construction: active validation is overwhelmingly plain text and has no control validation batches.
2. Early stopping: stopping criterion ignores format following, memory recall, audio quality, and multimodal grounding.
3. Dataset mix source identity: active run config differs from `data/propagator_dataset_mix.json`; make the launched mix explicit and archived.
4. Posttrain generation: repeated identity and code-word rows cause shallow memorization and identity intrusion.
5. Reset mask discipline: `[SESSION]` is not a hard reset; add explicit state-leak tests.
6. Chunk-boundary recall: test whether instructions before a boundary affect answer after the boundary.
7. Target masks: verify user-side text/audio is not accidentally supervised as assistant output.
8. `[TEXT_OUT]`/`[AUDIO_OUT]`/`[HYBRID_OUT]` target placement: verify exact modality token target after `[MODEL]`.
9. Audio task directionality: ASR/TTS rows must not be mixed into ambiguous audio/text continuation.
10. Audio codebook label shift: verify q0 main token and q1-q7 aux labels align to the same frame unless a delay pattern is intentionally implemented.
11. Eval sampler coverage: ensure every requested task has nonzero validation samples.
12. Candidate/generation head behavior: samples should distinguish forced runtime protocol from raw model decisions.

## 16. Prioritized implementation plan

### Immediate P0

1. Add `sample_05_format_following` fixture and protocol/token/mask regression test.
2. Add validation buckets for strict format, matrix recall, state reset/leakage, ASR, TTS, and hybrid.
3. Disable or replace CE-only early stopping until composite validation exists.
4. Replace repetitive posttrain rows with diverse instruction/format tasks.
5. Log effective examples/tokens/audio frames per source, task, and modality after packing.
6. Archive the exact launched dataset mix and cache metadata in the output directory.

### Next P1

1. Add memory diagnostics: memory norm, update norm, eta, forget, read-value norm, per-group key utilization.
2. Add chunk-boundary recall curriculum.
3. Add audio-specific metrics and fixed probes.
4. Rebuild dataset sampler with per-task weights and caps.
5. Add paired text/audio integrity checks.

### Later P2

1. Add image protocol and precomputed visual token training.
2. Benchmark tiny vision encoders on SL2610-class target runtime.
3. Distill a compact dense student.
4. Quantize and profile batch-1 streaming serving.
5. Add DPO only after SFT metrics are strong.

## 17. Code-change checklist

Recommended concrete changes:

- Add `data/regression/sample_05_format_following.jsonl`.
- Add `scripts/audit_protocol_fixture.py` or extend `scripts/smoke_training.py` to inspect token streams and masks.
- Add a deterministic generation regression script for fixed samples.
- Add `validation_control_batches > 0` by default.
- Add named validation buckets and per-bucket metrics.
- Replace CE-only early stopping with composite task score.
- Add dataset cache summary logs: source, task, modality, examples, chunks, text tokens, audio frames, image examples.
- Add hard warnings when a validation bucket has zero samples.
- Add state reset/leak tests for `[SESSION]` and sampler reset masks.
- Add chunk-boundary recall tests with fixed stream IDs.
- Add memory diagnostics to training logs.
- Add per-audio-codebook metrics and TTS fixed-probe decode.
- Add dataset mix v2 with the redesigned weights above.
- Add image special token `[IMAGE_IN]` and precomputed visual-token loader.
- Add edge memory report fields for activation peak, audio buffers, image encoder buffers, and NPU fallback ops.

## 18. Bottom line

The current model is not blocked by a simple "train longer" issue. Training longer under the same validation and mixture would likely improve plain continuation and shallow protocol behavior while leaving format following, matrix-memory recall, and speech grounding weak.

The next run should first fix validation, data mixture, regression tests, and memory diagnostics. Only then is it meaningful to compare architecture scale, longer unrolls, audio loss weights, or edge compression settings.

## 19. Safe fixes implemented

The repository now has a lightweight regression entrypoint:

```bash
./.venv/bin/python scripts/audit_prop_regressions.py
```

It does not train. It verifies protocol construction, synthetic audio-codebook frame alignment, recurrent-state source invariants, current generated sample failure, run metrics, repetitive posttrain data, configured mix weights, and cache imbalance.
It also instantiates a tiny random Propagator model to verify recurrent memory initialization, reset-mask behavior, valid-mask no-write behavior, and carry-vs-reset divergence.

Training-loop validation status:

- Control-validation metrics are included in `metric_sums`.
- `validation_composite_score` is written into `validation_metrics.json` and `metrics.jsonl` for new runs.
- `val_composite_score.png` is written at eval steps.
- Early stopping now uses `validation_composite_score` with the existing `early_stopping_min_delta` and `early_stopping_patience`.
- `duplex_task_acc=NaN` is treated as a zero duplex score in the composite instead of being ignored.

The new balanced mix is now the regular training default:

```bash
scripts/train.sh
```

To override it explicitly:

```bash
DATASET_MIX_FILE=data/propagator_dataset_mix.json scripts/train.sh
```

This does not fix the already-trained checkpoint. It prevents the same failure pattern from being invisible in future runs and gives the next run a better candidate mixture.

The old `data/propagator_posttrain_10k.jsonl` remains in the repository for reference, but balanced v2 no longer samples it. The audit script still reports its repetition because it explains the existing checkpoint behavior.

Remaining required work:

- add per-source/per-task validation buckets to training, not only an offline audit script;
- initialize real Mimi audio token ranges in the audio alignment audit so q0-q7 labels are checked against actual codec token IDs;
- add image `[IMAGE_IN]` token and precomputed visual-token loader;
- `[IMAGE_IN]` is now implemented; precomputed visual-token loader is still required;
- run a short dry-run cache build with the balanced mix before launching a full training run.
