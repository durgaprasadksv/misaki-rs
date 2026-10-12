# Parity with misaki 0.9.4

This fork exists to make misaki-rs produce **byte-identical** output to
**misaki 0.9.4**, the G2P that Kokoro v1.0 was trained against, so a Rust
serving binary can replace a Python one without changing pronunciation or word
timestamps.

## Why differential testing, and not replaying upstream commits

Two facts settle the method:

1. **misaki 0.9.4 is the end of the line.** It is the latest PyPI release
   (2025-04-05) *and* `hexgrad/misaki` HEAD still reads
   `__version__ = '0.9.4'`. Only three commits exist after the release — Python
   3.13 support, a pip dependency fix, and an optional transformer fallback —
   and none bump the version. There is no forward history to catch up on.

2. **The divergences are not version drift.** Of the six found so far, five
   were never any misaki version's behaviour; they are shortcuts and
   inventions in the port:

   | divergence | drift, or port artefact? |
   |---|---|
   | lexicons from a different release | **drift** |
   | missing `ɾ`→`T`, `ʔ`→`t` pass | port omission (it is in 0.9.4's `en.py`) |
   | `whitespace` hardcoded to `" "` | port bug — no misaki ever did this |
   | `future_vowel` from spelling (`"aeiou"`) | port shortcut, self-labelled "simple heuristic" |
   | char-by-char spell-out for OOV words | port invention |
   | punctuation reduced to `;:,.!?` | port shortcut |

   A commit replay would have found exactly one of these.

So misaki 0.9.4 is used as an **oracle** instead: run it over a corpus, freeze
the output, assert against it from Rust. That also covers the rare paths a
changelog would never mention.

## The gate

```sh
cargo test --no-default-features --test parity -- --nocapture
```

`--no-default-features` is required: the fixtures were generated with
`fallback=None`, so espeak must be off to compare like with like.

Two properties are asserted per case:

* the **phoneme string** — what the model is fed;
* the **per-token `(text, phonemes)` pairs** — Kokoro's `pred_dur` has one entry
  per input token, so word→phoneme attribution is what the **word timestamps**
  are derived from. A suite checking only the string could pass with the
  timestamps silently wrong.

Fixtures are regenerated with `tools/gen_fixtures.py` against a venv holding
misaki 0.9.4; `tools/build_corpus.py` assembles the corpora. No Python is
needed to *run* the tests.

## Progress

| | adversarial (71) | prose (700) |
|---|---|---|
| as forked | 0% | 0% |
| + 0.9.4 lexicons, flap pass, real whitespace | 49.3% | 40.3% |
| + `token_context` ported | 52.1% | 41.0% |
| + OOV stays unresolved, PUNCTS ported | **53.5%** | **57.1%** |

Target is 100% on both. The adversarial corpus is deliberately hostile
(heteronyms, acronyms, dates, currency, quotes) and is not representative of
real text; report both numbers, because an implementation can score well on
prose while failing every heteronym — the failure a listener notices most.

## Known remaining gaps

In rough order of how many cases they account for:

1. **Acronyms.** `get_NNP` is not wired: `PDFs` → `pˈi dˈi ˈɛf ˈɛs` where misaki
   gives `pˌidˌiˈɛfs` (joined, secondary stresses, plural `s` as a suffix rather
   than a spelled letter). `stem_s`/`append_s` exist; the NNP path that calls
   them does not.
2. **The POS tagger.** The embedded `PerceptronTagger` does not reproduce
   spaCy's tags. Two consequences: heteronyms resolve wrongly (`I read
   yesterday` → present tense), and punctuation is mis-tagged (`“` as `NNP`,
   `’` as `VBP`), which is why `PUNCT_TAG_PHONEMES` had to be approached by
   character here rather than by tag. Exposure on heteronyms is **bounded**:
   only 790 of 90,201 gold entries are POS-conditional.
3. **The multi-token word layer.** `fold_left`, `retokenize`, `merge_tokens`,
   `resolve_tokens`, `stress_weight`, `make_subtokenize_once` are all absent.
   `resolve_tokens` is what rebalances stress across a multi-token word, which
   is behind the residual `ˈ`/`ˌ` differences (`in` → `ˈɪn` vs `ɪn`).
   Contractions also mis-split: `It’s` becomes `It` + `’` + `s`.
4. **Numbers, currency, ordinals.** `is_currency`, `get_number`,
   `append_currency`, `numeric_if_needed` are absent or partial. `num2words` is
   already a dependency, so the primitive is available.
