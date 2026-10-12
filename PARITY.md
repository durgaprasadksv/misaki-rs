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
| + OOV unresolved, PUNCTS ported | 53.5% | 57.1% |
| + number forms, adjacency merge, abbreviations | 62.0% | 63.9% |
| + apostrophe normalisation, acronym trigger | 60.6% | 74.7% |
| + stemmers keep case | 66.2% | 74.7% |
| + capitalisation stress, stress rebalancing | 64.8% | 81.1% |
| + prespace guard, ordinals | 67.6% | 81.1% |
| + underscore merges, opening quote stands alone | 67.6% | 83.9% |
| + multi-token span loop | **67.6%** | **84.7%** |

Phoneme strings alone match on 87.7% of prose; the gate is the lower of that
and per-token attribution, because `pred_dur` is per token.

Target is 100% on both.

## What is left, measured rather than guessed

On the 623 prose cases where **tokenization already agrees**:

| | |
|---|---|
| phonemes match | 593 |
| fail, and the POS tag also differs | **64** |
| fail with every tag correct | **9** |

So the ported logic is essentially right, and the dominant remaining cause is
the **POS tagger**. Its agreement with spaCy is **89-90%**, and the
disagreements that change pronunciation are exactly the ones misaki keys on:

| spaCy -> ours | n | effect |
|---|---|---|
| `DT` -> `NNP` on *A* | 8 | `ɐ` vs `ˈA` -- `get_special_case` branches on `tag == 'DT'` |
| `DT`/`IN` confusion on *that* | 21 | `ðˈæt` vs `ðæt` (a POS-conditional lexicon entry) |
| `RB`/`RP` -> `IN` on *in* | 6 | stress |
| `VBD` -> `NN` on *dove*, *read* | — | the heteronyms |

### Reaching 100% means shipping spaCy's tagger

misaki's pronunciation is POS-conditional (790 of 90,201 gold entries are
tag-keyed, plus the `get_special_case` branches), so exact parity requires
exactly spaCy's tags. The embedded `PerceptronTagger` cannot get there; it is a
different model class.

`en_core_web_sm`'s tagger is small and fully specified, which makes this
bounded work rather than open-ended:

```
MultiHashEmbed   6 hash embeddings (NORM/PREFIX/SUFFIX/SHAPE/...) -> 576
                 -> maxout -> layernorm -> 96
MaxoutWindowEncoder  4x residual(expand_window(3) -> maxout -> layernorm)
Softmax          96 -> 50 tags
```

**1,565,664 parameters, 6.3 MB fp32** — smaller than the lexicons already
vendored here. Every operation is elementary (hash lookup, affine, maxout as a
reshape-and-max, layernorm, residual, window concat); the only fidelity risk is
reproducing Thinc's MurmurHash seeding exactly. No ONNX runtime is required,
though a Kokoro service already carries one.

Projected: fixing the tagger should convert most of those 64, giving roughly
94%. The remaining ~6% is the 77 cases where tokenization still disagrees plus
the 9 residual logic failures — both visible to the gate and chaseable the same
way everything above was.

## Known remaining gaps, in order

1. **POS tagger** — 64 of 73 wrong-phoneme tokens. See above.
2. **Tokenization** — 77 of 700 prose cases still differ in token count.
   Remaining shapes are rarer: bracketed citations, stacked quotes, `17—.`
3. **Currency** — `append_currency` is unwired, so `$9` gives `nˈIn` and not
   `nˈIn dˈɑləɹz`. `is_currency`/`numeric_if_needed` are absent.
4. **Negative numbers** — misaki renders `-40` as the empty string; we read it
   as "forty".
5. **`grow_dictionary`** — absent; nothing in our pipeline calls it.
