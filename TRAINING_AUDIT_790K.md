# Training Audit at Step 790,000

## Decision

Do not continue the v2 checkpoint as the main training line. Preserve step 780,000 as a recovery and comparison artifact, but start the corrected v3 run from new model and optimizer state because the default audio vocabulary changes from EnCodec to Mimi.

## Convergence Evidence

The run had 166 evaluations through step 795,000. Important best values occurred well before the latest step and then regressed:

| Metric | Best step | Best | Step 790,000 |
| --- | ---: | ---: | ---: |
| Validation loss | 655,000 | 34.9215 | 41.3842 |
| Text token accuracy | 455,000 | 0.7076 | 0.6259 |
| ASR task accuracy | 385,000 | 0.3977 | 0.3203 |
| TTS task accuracy | 190,000 | 0.4277 | 0.3670 |
| Audio q0 accuracy | 500,000 | 0.4428 | 0.3826 |
| Audio auxiliary token accuracy | 675,000 | 0.1994 | 0.1479 |

At step 790,000, main CE was 1.9813 while reported auxiliary audio CE was 29.9615. Exact all-codebook frame accuracy was 0.0208. This is not a normally converging run.

## Root Causes

1. The 30-epoch schedule produced 14,320,860 optimizer steps. At step 780,000 the old cosine schedule was still at about 99.3% of the 3e-4 peak learning rate.
2. Cached source sizes overrode configured mixture weights. FineWeb, Wikipedia, and UltraChat were heavily overrepresented while identity and several speech sources were underrepresented.
3. Validation sampled different streams at every evaluation, making curves noisy and comparisons unreliable.
4. Synthetic controls paired prompts and responses by unrelated modular indices. The name prompt was directly trained against a generic non-identity answer.
5. The auxiliary loss summed seven residual codebook losses and then multiplied the result by 3.
6. Training represented eight codebooks as one frame, but audio generation advanced recurrent memory once per codebook and used the main vocabulary head for all codebooks.
7. Residual codebooks were predicted independently without conditioning on q0.
8. Batched variable-length audio retained zero-padding as codec frames for shorter EchoX turns.
9. FineWeb and Wikipedia rows were all converted into answers to the same `Continue this stream.` prompt.
10. EnCodec is a 75 Hz reconstruction codec without a speech-semantic first codebook. This creates long sequences and a poor representation for joint ASR/TTS learning.
11. The initial memory forget setting implied roughly a 69-step half-life, less than one second at 75 Hz, while truncated backpropagation covered only 32 frames.
12. Candidate-head evaluation excluded roughly half of the 16k text tokenizer vocabulary.

## Corrected v3 Direction

- Mimi at 24 kHz, 12.5 Hz, 8 codebooks, and 2048 entries per codebook.
- One recurrent update per synchronized audio frame in both training and inference.
- q1 through q7 conditioned on q0 and predicted by their trained auxiliary heads.
- Source-weighted stateful sampling and fixed validation streams.
- Plain-text context/continuation preprocessing instead of one repeated instruction.
- Peak LR 1e-4, 5k warmup, 0.01 weight decay, 1.2M absolute-step cap, 3 epochs only as a fallback ceiling, and fixed-validation early stopping.
- Lower memory write/forget rates and normalized multi-codebook input embeddings.
- Per-task CE for text, ASR, TTS, and duplex validation.

## Checkpoint and GCS State

- Step 790,000 was evaluation-only under the old 20k checkpoint cadence.
- The latest actual recovery checkpoint at audit time was step 780,000.
- Step 780,000 checkpoint and step 790,000 evaluation artifacts were synced under `gs://propagator-gde-project-aicloud/propagator-duplex/`.
- The corrected run uses a separate `propagator-multimodal` output and `propagator-duplex` GCS prefix.

## Verification

- `scripts/smoke_training.py` checks source weighting, control examples, plain-text continuation, multimodal loss, q0-conditioned auxiliary heads, and frame-consistent generation.
- `scripts/smoke_audio_preprocessing.py` runs real Mimi and EnCodec encode/decode tests with unequal waveform lengths and verifies frame trimming and codebook ranges.
