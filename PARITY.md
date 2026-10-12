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
| 0.9.4 lexicons, flap pass, real whitespace | 49.3% | 40.3% |
| `token_context` ported | 52.1% | 41.0% |
| OOV unresolved, PUNCTS ported | 53.5% | 57.1% |
| number forms, adjacency merge, abbreviations | 62.0% | 63.9% |
| apostrophe normalisation, acronym trigger | 60.6% | 74.7% |
| stemmers keep case | 66.2% | 74.7% |
| capitalisation stress, stress rebalancing | 64.8% | 81.1% |
| prespace guard, ordinals | 67.6% | 81.1% |
| underscore merges, opening quote stands alone | 67.6% | 83.9% |
| multi-token span loop | 67.6% | 84.7% |
| **ported spaCy tagger, word-level tagging** | 70.4% | 88.9% |
| punctuation before lexicon, closing quotes | 70.4% | 90.0% |
| `lookup` spells out non-NNP unknowns | 76.1% | 90.7% |
| ordinals reach the number path | 77.5% | 90.9% |
| currency, negative numbers | **78.9%** | **90.9%** |

Prose phoneme strings alone match on 94.1%; the gate takes the lower of that and
per-token attribution. The adversarial corpus is deliberately hostile
(heteronyms, acronyms, dates, currency, stacked quotes) and is not
representative of real text — report both, because an implementation can score
well on prose while failing every heteronym, the error a listener notices most.

Target is 100% on both.

## The POS tagger: done, exactly

`src/spacy_tagger.rs` ports `en_core_web_sm`'s tagger and reproduces it
**exactly — 20,623/20,623 tokens, 771/771 sentences** (`tests/tagger_parity.rs`,
which also checks the six feature columns separately so a mismatch can be
attributed to the features or the network).

This mattered because misaki's pronunciation is POS-conditional: 790 of the
90,201 gold entries are tag-keyed and `get_special_case` branches on the tag
directly (`A` as `DT` is `ɐ`, as `NNP` it is `ˈA`). The perceptron tagger it
shipped with agreed on ~89% and accounted for 64 of the 73 wrong-phoneme tokens
remaining at the time.

Four details cost real accuracy and none are visible in the architecture:

* **Thinc's HashEmbed hash is not generic MurmurHash3.** It is hand-specialised
  for a single `u64` (`numpy_ops.pyx`, `MurmurHash3_x86_128_uint64`), ~25 lines;
  a stock murmur3 crate does not reproduce it.
* **spaCy's StringStore does not hash every string.** It holds 457 symbols with
  fixed small IDs — the shape `X` is 101 — so a one-character uppercase token's
  SHAPE feature is 101, not its hash. `src/symbols.rs` carries the table. ~0.7%.
* **`with_array(pad=4)`** wraps the encoder input in four zero rows per side,
  then strips them. Not the same as zero-padding inside `expand_window`: the pad
  rows pass through maxout and layernorm, so from the second block on a boundary
  token's neighbour is the layernorm bias, not zero. Omitting it held agreement
  at 99.0% and is why `Shhh!` tagged `NNP` instead of `UH`.
* **`word_shape` tests `char.isalpha()` before case**, so a caseless alphabetic
  character is `x` rather than itself (`ˈ` is alpha, neither upper nor lower);
  and 100+ characters shape to the literal `LONG`.

Each was found by bisecting the forward pass against spaCy's intermediate
outputs, after a numpy reimplementation reproduced it exactly.

## What is left

Almost all of it is one root cause: **misaki groups subtokens into words by
`tag in PUNCT_TAGS` on spaCy's OWN tokens, and we do not reproduce spaCy's
tokenizer.** The grouping here approximates that with character tests, which is
right most of the time and demonstrably not always:

| | spaCy's tag | misaki | ours |
|---|---|---|---|
| `_` in `_like` | `UH` | merges -> `_like` | merges ✓ |
| `_` in `_it` | `NFP` | stands alone | merges ✗ |
| `'` in `dogs'` | `POS` | merges -> `dogs'` | stands alone ✗ |
| `’` in `Thomas’` | `''` | stands alone | stands alone ✓ |

Structurally identical inputs, different tags, different grouping. No character
rule can separate them — the tag is the rule.

The tagger is already exact, so the missing piece is **spaCy's tokenizer**:
**1,347 tokenizer exceptions** plus 120 prefix / 83 suffix / 8 infix patterns
and a URL matcher. All of it is data; the delicate part is the peeling algorithm
that consults the exception table at each stage. `src/abbrev.rs` already carries
the 146 `.`-terminated exceptions, which is the slice that mattered most.

Smaller, independent remainders:

1. **The `2` -> `to` alias.** misaki rewrites a `2` between letters:
   `elif 0 < j < len(tks)-1 and tk.text == '2' and (tks[j-1].text[-1]+tks[j+1].text[0]).isalpha()`,
   so `G2P` is `ʤˈitəpˈi`. The alias also makes the token stand alone.
2. **Times.** `5:30` and `3:42:15` keep their colons where misaki drops them.
3. **Straight quotes by position.** A leading `'` is `“` and a trailing one `”`;
   spaCy decides it by tag, and we render both as `”`.
4. **Decimals in context** (`26.2`) and phone-number runs (`555-1234`).
5. **`grow_dictionary`** remains absent; nothing in our pipeline calls it.
