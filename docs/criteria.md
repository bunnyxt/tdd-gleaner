# Inclusion and annotation criteria

These criteria guide the first annotation pilot and model evaluation. The goal is to collect voicebank performances, prioritizing low false-negative rates for popular eligible works.

## Two independent judgments

| Field | Meaning |
| --- | --- |
| `uses_voicebank` | The work actually uses a virtual singer voicebank to perform vocals or speech, rather than merely mentioning a voicebank, character, or engine. |
| `is_vocal_work` | The video primarily presents a vocal performance, including songs, comic dialogue, or tuned speech. News and merchandise presentations are outside this scope. |

Normal automated inclusion requires both fields to be `true`. A human-performed song can satisfy the second criterion without satisfying the first. An unresolved human label may temporarily be `null` rather than being forced to `false`; this is an annotation status, not a third boolean value for completed model judgments.

## Typical cases

| Video content | Normal inclusion decision |
| --- | --- |
| An original song or cover performed using a virtual singer voicebank | Include |
| A different virtual singer voicebank covering an existing virtual singer song | Include |
| Comic dialogue or tuned speech performed using a voicebank | Include |
| A human cover of a virtual singer song | Exclude |
| Virtual singer news or merchandise presentations, including brief preview clips | Exclude |
| Fully AI-generated music without a corresponding voicebank | Exclude |

The original song, character names in the title, upload category, or discovery source cannot independently establish eligibility. The use of AI technology does not replace the two content judgments. Do not exclude new voicebanks through a closed list of supported engines.

## Model input

Version 1 includes only `title`, `desc`, and `tags`. Uploader nicknames, view counts, discovery sources, request metadata, human labels, and annotation rationales are not model inputs.

Do not invoke the model when information acquisition fails. Record a separate acquisition failure or unavailable status. Distinguish a legitimately empty description from an acquisition failure; a deleted video must not become two `false` labels.

Human annotators may watch videos, listen to audio, and verify voicebank information. Note insufficient textual evidence where applicable, so evaluation can distinguish classification errors from missing input information. Leave unresolved cases, such as mixed human and voicebank performances or compilations, pending until the pilot clarifies the criteria.

## Judgment and execution

Store model outputs separately from inclusion execution. When a model returns probabilities, convert them into flags using validated thresholds; do not assume every model should share a threshold. Correct output structure does not guarantee correct classification.

Persistent administrator overrides can force inclusion or exclusion. Ordinary inclusion requests and rechecks cannot supersede them. A one-time bypass of content checks is not a persistent override.

Removal uses a separate tracking switch and preserves historical data. Acquisition failures, inference failures, or ordinary title and tag changes should not directly trigger removal. Text alone cannot reliably establish whether the underlying audio or video has been replaced.
