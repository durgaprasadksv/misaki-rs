use crate::fallback::{Fallback, FallbackError};
use crate::language::Language;
#[cfg(feature = "espeak")]
use crate::fallback::EspeakFallback;
use crate::languages::{LanguageRules, english::English};
use crate::lexicon::Lexicon;
use thiserror::Error;
use crate::tagger::PerceptronTagger;
use crate::token::MToken;
use num2words::Num2Words;
use regex::Regex;
use std::collections::HashMap;

#[derive(Error, Debug)]
pub enum G2PError {
    #[error("fallback error: {0}")]
    Fallback(#[from] FallbackError),
}

/// A per-word override parsed from `[text](feature)` markdown-link syntax.
///
/// Matches the set of features accepted by Python misaki's
/// `Lexicon.preprocess` in `misaki/en.py`. The four shapes are:
///
/// * `[hello](+2)` / `[hello](-1)` / `[hello](2)` — integer stress
/// * `[hello](0.5)` / `(+0.5)` / `(-0.5)` — half-stress (these literals only)
/// * `[xyzzy](/həˈloʊ/)` — direct phoneme override
/// * `[5](#cardinal#)` — number-formatting flags
///
/// Anything else (including `=` or plain words) is dropped silently, matching
/// the upstream `else: f = None` branch.
#[derive(Debug, Clone, PartialEq)]
pub enum LinkFeature {
    Stress(f64),
    Phonemes(String),
    NumFlags(String),
}

/// A parsed `[text](feature)` marker, anchored to byte offsets in the
/// stripped text returned by [`G2P::preprocess_links`].
#[derive(Debug, Clone)]
pub struct LinkMarker {
    pub byte_start: usize,
    pub byte_end: usize,
    pub feature: LinkFeature,
}

fn parse_link_feature(raw: &str) -> Option<LinkFeature> {
    // Mirrors the feature-parsing ladder in misaki/en.py::Lexicon.preprocess.
    // No trim — the inner text is consumed verbatim.
    if raw.is_empty() {
        return None;
    }

    // Integer stress: optional leading + or -, then all-ASCII-digit body.
    // Python: `is_digit(f[1 if f[:1] in ('-','+') else 0:])`.
    let body = match raw.as_bytes()[0] {
        b'+' | b'-' => &raw[1..],
        _ => raw,
    };
    if !body.is_empty() && body.bytes().all(|b| b.is_ascii_digit()) {
        if let Ok(n) = raw.parse::<i32>() {
            return Some(LinkFeature::Stress(n as f64));
        }
    }

    // Half stress: only the three literals Python recognises.
    match raw {
        "0.5" | "+0.5" => return Some(LinkFeature::Stress(0.5)),
        "-0.5" => return Some(LinkFeature::Stress(-0.5)),
        _ => {}
    }

    // /phonemes/ — Python strips one leading slash and all trailing slashes
    // (via `f[1:].rstrip('/')` at parse and `v.lstrip('/')` at apply time).
    if raw.len() > 1 && raw.starts_with('/') && raw.ends_with('/') {
        let inner = raw[1..].trim_end_matches('/');
        return Some(LinkFeature::Phonemes(inner.to_string()));
    }

    // #num_flags# — same lstrip/rstrip semantics as the phoneme branch.
    if raw.len() > 1 && raw.starts_with('#') && raw.ends_with('#') {
        let inner = raw[1..].trim_end_matches('#');
        return Some(LinkFeature::NumFlags(inner.to_string()));
    }

    None
}

pub struct G2P {
    pub lexicon: Lexicon,
    pub unk: String,
    subtoken_regex: Regex,
    link_regex: Regex,
    /// The perceptron tagger this crate shipped with. No longer used for
    /// pronunciation -- `spacy_tagger` is, because misaki's output depends on
    /// spaCy's exact tags -- but kept so the two remain comparable.
    #[allow(dead_code)]
    tagger: PerceptronTagger,
    spacy_tagger: crate::spacy_tagger::Tagger,
    rules: Box<dyn LanguageRules>,
    fallback: Option<Box<dyn Fallback>>,
}

impl G2P {
    pub fn new(lang: Language) -> Self {
        // misaki's SUBTOKEN_REGEX (`en.py:55`), alternative for alternative and
        // in the same order, since regex alternation takes the first match:
        //
        //   ^['‘’]+ | \p{Lu}(?=\p{Lu}\p{Ll}) | (?:^-)?(?:\d?[,.]?\d)+ | [-_]+
        //   | ['‘’]{2,} | \p{L}*?(?:['‘’]\p{L})*?\p{Ll}(?=\p{Lu})
        //   | \p{L}+(?:['‘’]\p{L})* | [^-_\p{L}'‘’\d] | ['‘’]+$
        //
        // Two bugs were in the character classes here. The word-internal
        // apostrophe class read `[''']` -- three copies of the STRAIGHT
        // apostrophe where misaki has straight plus both curly -- and the
        // catch-all excluded only the straight one. So a curly ’ fell through
        // to the catch-all and became its own token: `man’s` split into
        // `man`,`’`,`s` while `man's` stayed whole. Every curly possessive and
        // contraction in real prose came apart -- `man’s`, `Alice’s`,
        // `hadn’t`, `won’t`, `Raskolnikov’s`.
        //
        // The repeat is `\p{L}` and not `\p{L}+`, also as misaki has it; that is
        // what makes `o’clock` subtokenize as `o’c` + `lock`.
        //
        // The two alternatives with lookahead (`\p{Lu}(?=\p{Lu}\p{Ll})` and the
        // camel-case split) need `fancy_regex`; they are handled separately
        // below so this stays on the fast `regex` engine.
        let subtoken_regex = Regex::new(
            r"(?x)
            ^['‘’]+ |
            (?:^-)?(?:\d?[,.]?\d)+ |
            [\-_]+ |
            ['‘’]{2,} |
            \p{L}+(?:['‘’]\p{L})* |
            [^\-_\p{L}'‘’\d] |
            ['‘’]+$
        ",
        )
        .unwrap();

        let weights_json = include_str!("resources/tagger/weights.json");
        let classes_txt = include_str!("resources/tagger/classes.txt");
        let tags_json = include_str!("resources/tagger/tags.json");

        let rules: Box<dyn LanguageRules> = match lang {
            Language::EnglishUS | Language::EnglishGB => Box::new(English),
            // Language::Italian => Box::new(Italian),
        };

        #[cfg(feature = "espeak")]
        let fallback: Option<Box<dyn Fallback>> = match EspeakFallback::new(lang == Language::EnglishGB) {
            Ok(fb) => Some(Box::new(fb)),
            Err(e) => {
                log::warn!("espeak-ng fallback unavailable: {}", e);
                None
            }
        };
        #[cfg(not(feature = "espeak"))]
        let fallback: Option<Box<dyn Fallback>> = None;

        // Mirrors `LINK_REGEX = re.compile(r'\[([^\]]+)\]\(([^\)]*)\)')`
        // from misaki/en.py.
        let link_regex = Regex::new(r"\[([^\]]+)\]\(([^\)]*)\)").unwrap();

        Self {
            lexicon: Lexicon::new(lang),
            unk: "❓".to_string(),
            subtoken_regex,
            link_regex,
            tagger: PerceptronTagger::new(weights_json, classes_txt, tags_json),
            // The ported spaCy tagger. Kept alongside the perceptron rather
            // than replacing it so the old one stays available for comparison.
            spacy_tagger: crate::spacy_tagger::Tagger::load(),
            rules,
            fallback,
        }
    }

    pub fn with_fallback(lang: Language, fallback: Option<Box<dyn Fallback>>) -> Self {
        let mut g2p = Self::new(lang);
        g2p.fallback = fallback;
        g2p
    }

    /// Legacy preprocess shape, kept for API compatibility.
    ///
    /// The third return value (token-index → feature) is intentionally empty
    /// — markers are now exposed via [`G2P::preprocess_links`] using byte
    /// offsets, which compose with our subtokenizer (Python misaki keys
    /// features by spaCy-tokenized index; we don't have spaCy).
    pub fn preprocess(&self, text: &str) -> (String, Vec<String>, HashMap<usize, String>) {
        let (stripped, _) = self.preprocess_links(text);
        let tokens: Vec<String> = stripped.split_whitespace().map(|s| s.to_string()).collect();
        (stripped, tokens, HashMap::new())
    }

    /// Parse `[text](feature)` markers from `text` and return the stripped
    /// text along with markers anchored to byte ranges in the stripped text.
    ///
    /// Mirrors the parsing half of `Lexicon.preprocess` in misaki/en.py.
    pub fn preprocess_links(&self, text: &str) -> (String, Vec<LinkMarker>) {
        let mut stripped = String::with_capacity(text.len());
        let mut markers: Vec<LinkMarker> = Vec::new();
        let mut last_end = 0usize;
        for caps in self.link_regex.captures_iter(text) {
            let whole = caps.get(0).unwrap();
            let inner = caps.get(1).unwrap().as_str();
            let feat_str = caps.get(2).unwrap().as_str();
            stripped.push_str(&text[last_end..whole.start()]);
            let start = stripped.len();
            stripped.push_str(inner);
            let end = stripped.len();
            if let Some(feature) = parse_link_feature(feat_str) {
                markers.push(LinkMarker {
                    byte_start: start,
                    byte_end: end,
                    feature,
                });
            }
            last_end = whole.end();
        }
        stripped.push_str(&text[last_end..]);
        (stripped, markers)
    }

    /// misaki's punctuation test (`retokenize`):
    ///
    ///     elif tk.tag in PUNCT_TAGS and not all(97 <= ord(c.lower()) <= 122 for c in tk.text):
    ///
    /// with PUNCT_TAGS = {".", ",", "-LRB-", "-RRB-", "``", "\"\"", "''", ":",
    /// "$", "#", "NFP"}. Such a token gets its phonemes in `retokenize` and is
    /// therefore appended STANDALONE, never merged into a neighbouring word.
    ///
    /// This is the rule, and no character test substitutes for it: spaCy tags
    /// the `_` in `_like` as UH and the one in `_it` as NFP, so misaki merges
    /// the first and not the second from structurally identical input. Likewise
    /// `dogs'` has tag POS and merges, while `Thomas’` has `''` and does not.
    fn is_punct_token(tk: &MToken) -> bool {
        const PUNCT_TAGS: [&str; 11] = [
            ".", ",", "-LRB-", "-RRB-", "``", "\"\"", "''", ":", "$", "#", "NFP",
        ];
        if !PUNCT_TAGS.contains(&tk.tag.as_str()) {
            return false;
        }
        // ... and not entirely a-z, so a word that happens to carry a
        // punctuation tag is still a word.
        !tk.text.chars().all(|c| {
            let l = c.to_lowercase().next().unwrap_or(c);
            ('a'..='z').contains(&l)
        })
    }

    /// Subtoken runs that spaCy would have produced as single tokens.
    ///
    /// Only the period matters: our subtokenizer always splits it, while spaCy
    /// keeps its 146 known abbreviations whole (`Mr.`, `a.m.`, and single
    /// initials like `K.`). Everything else our tokenizer already agrees on.
    fn group_for_tagging(tokens: &[MToken]) -> Vec<std::ops::Range<usize>> {
        let mut out: Vec<std::ops::Range<usize>> = Vec::new();
        let mut i = 0usize;
        while i < tokens.len() {
            let mut end = i + 1;
            // `Mr` + `.` -> `Mr.`, but only while the pieces are adjacent.
            if i + 1 < tokens.len()
                && tokens[i].whitespace.is_empty()
                && tokens[i + 1].text == "."
            {
                let cand = format!("{}.", tokens[i].text);
                if crate::abbrev::is_period_abbreviation(&cand) {
                    end = i + 2;
                }
            }
            // `’` + `s` -> `’s`. spaCy emits that as one token and tags it POS,
            // which is not a PUNCT_TAG, so misaki merges it into the word.
            // Tagging a lone `’` instead yields `''` and splits `Lydia’s` apart.
            if end == i + 1 && tokens[i].text.chars().count() == 1 {
                let c = tokens[i].text.chars().next().unwrap();
                if matches!(c, '\'' | '\u{2018}' | '\u{2019}') {
                    let mut cand = tokens[i].text.clone();
                    let mut j = i;
                    while j + 1 < tokens.len() && tokens[j].whitespace.is_empty() {
                        cand.push_str(&tokens[j + 1].text);
                        j += 1;
                        if crate::abbrev::is_contraction_piece(&cand) {
                            end = j + 1;
                            break;
                        }
                        if cand.chars().count() > 6 {
                            break;
                        }
                    }
                }
            }
            out.push(i..end);
            i = end;
        }
        out
    }

    /// Port of misaki's module-level `merge_tokens` (`en.py`).
    ///
    /// Collapses a run of adjacent subtokens into the single word misaki would
    /// have produced. The text is rebuilt with the interior whitespace put
    /// back, the group inherits the LAST token's trailing whitespace, and the
    /// tag comes from the "heaviest" member -- misaki scores each token's text
    /// as `sum(1 if c == c.lower() else 2 for c in text)` and takes the
    /// maximum, so `five-storied` takes `storied`'s JJ (7 beats 4 and 1) and
    /// `It’s` takes `It`'s PRP (3 beats 2).
    fn merge_text_tag(group: &[MToken]) -> (String, String, String) {
        let mut text = String::new();
        for tk in &group[..group.len() - 1] {
            text.push_str(&tk.text);
            text.push_str(&tk.whitespace);
        }
        text.push_str(&group[group.len() - 1].text);

        let weight = |t: &str| -> usize {
            t.chars()
                .map(|c| if c.to_lowercase().next() == Some(c) { 1 } else { 2 })
                .sum()
        };
        let tag = group
            .iter()
            .max_by_key(|tk| weight(&tk.text))
            .map(|tk| tk.tag.clone())
            .unwrap_or_default();

        let ws = group[group.len() - 1].whitespace.clone();
        (text, tag, ws)
    }

    /// Port of the grouping half of misaki's `G2P.retokenize` (`en.py`).
    ///
    /// misaki groups subtokens into words by ADJACENCY, not by character
    /// class: a token with no trailing whitespace opens a group, and each
    /// following token joins it while the group's last member still has no
    /// trailing whitespace.
    ///
    ///     elif words and isinstance(words[-1], list) and not words[-1][-1].whitespace:
    ///         tk._.is_head = False; words[-1].append(tk)
    ///     else:
    ///         words.append(tk if tk.whitespace else [tk])
    ///
    /// This is what lets a finer tokenizer than spaCy's stay correct: `Mr` `.`
    /// and `It` `’` `s` and `five` `-` `storied` are each adjacent runs, so they
    /// merge back into `Mr.`, `It’s` and `five-storied`. Returns index ranges
    /// into `tokens`.
    fn group_by_adjacency(tokens: &[MToken]) -> Vec<std::ops::Range<usize>> {
        // misaki's FIRST branch takes precedence over adjacency:
        //
        //     if tk._.alias is not None or tk.phonemes is not None:
        //         words.append(tk)
        //
        // and `retokenize` has already given punctuation its phonemes by then,
        // so a punctuation mark is always its own word. `is_punct_token` is
        // misaki's own test, on the TAG; the character approximation that used
        // to be here could not tell `_like` from `_it`.
        let is_standalone = |tk: &MToken, _is_last: bool| -> bool { Self::is_punct_token(tk) };

        let mut groups: Vec<std::ops::Range<usize>> = Vec::new();
        let mut open = false; // is the last group still extensible?
        for i in 0..tokens.len() {
            if is_standalone(&tokens[i], i + 1 == tokens.len()) {
                // ... unless joining it to the open group spells an
                // abbreviation spaCy would never have split. Our tokenizer
                // splits on the period unconditionally, so `Mr` + `.` must be
                // put back together; spaCy's own exception list decides.
                let rejoin = tokens[i].text == "." && open;
                if rejoin {
                    let g = groups.last().unwrap().clone();
                    let mut candidate = String::new();
                    for tk in &tokens[g.clone()] {
                        candidate.push_str(&tk.text);
                        candidate.push_str(&tk.whitespace);
                    }
                    candidate.push('.');
                    if crate::abbrev::is_period_abbreviation(candidate.trim()) {
                        let g = groups.last_mut().unwrap();
                        *g = g.start..i + 1;
                        open = tokens[i].whitespace.is_empty();
                        continue;
                    }
                }
                groups.push(i..i + 1);
                open = false;
                continue;
            }
            if open && tokens[groups.last().unwrap().end - 1].whitespace.is_empty() {
                let g = groups.last_mut().unwrap();
                *g = g.start..i + 1;
            } else {
                groups.push(i..i + 1);
                // A token that already carries trailing whitespace is a
                // finished word: misaki appends it bare rather than as a list,
                // and only a list can be extended.
                open = tokens[i].whitespace.is_empty();
            }
        }
        groups
    }

    /// misaki's lookup-key normalisation (`Lexicon.__call__`): curly single
    /// quotes become straight ones, then NFKC. The lexicons are keyed with
    /// straight apostrophes.
    ///
    /// NFKC itself is not applied here -- it would need a unicode-normalisation
    /// dependency, and the characters it folds that matter for English
    /// (ligatures, full-width forms) do not appear in the corpora. The quote
    /// substitution is the part that changes lookups.
    fn normalize_lookup(text: &str) -> String {
        text.replace('\u{2018}', "'").replace('\u{2019}', "'")
    }

    /// misaki's punctuation phonemes (`en.py:598-599`): the PUNCT_TAG_PHONEMES
    /// mapping, then keep only characters in PUNCTS.
    ///
    ///     tk.phonemes = PUNCT_TAG_PHONEMES.get(
    ///         tk.tag, ''.join(c for c in tk.text if c in PUNCTS))
    ///
    /// with PUNCTS = ';:,.!?\u2014\u2026"\u201c\u201d' and `` -> \u201c, '' -> \u201d, -LRB- -> (,
    /// -RRB- -> ). An empty result is correct for a symbol in no class (`*`).
    fn punct_phonemes(text: &str) -> String {
        const PUNCTS: &str = ";:,.!?\u{2014}\u{2026}\"\u{201c}\u{201d}";
        text.chars()
            .map(|c| match c {
                '\u{2018}' | '`' => '\u{201c}',
                '\u{2019}' | '\'' => '\u{201d}',
                '[' | '{' => '(',
                ']' | '}' => ')',
                other => other,
            })
            .filter(|c| PUNCTS.contains(*c) || matches!(c, '(' | ')'))
            .collect()
    }

    /// Port of misaki's `G2P.token_context` (`en.py`).
    ///
    /// Decides what the token to the LEFT should assume about what follows it:
    /// whether the next sound is a vowel (`the` -> `\u00f0i` vs `\u00f0\u0259`, `a` -> `\u0250`), and
    /// whether the next word is `to` (`to` -> `t\u028a` vs `t\u0259`). Both are decided
    /// from the RESOLVED PHONEMES, never from spelling.
    fn token_context(
        ctx: &crate::lexicon::TokenContext,
        ps: Option<&str>,
        tk: &MToken,
    ) -> crate::lexicon::TokenContext {
        const VOWELS: &str = "AIOQWYaiu\u{e6}\u{251}\u{252}\u{254}\u{259}\u{25b}\u{25c}\u{26a}\u{28a}\u{28c}\u{1d7b}";
        const CONSONANTS: &str = "bdfhjklmnpstvwz\u{f0}\u{14b}\u{261}\u{279}\u{27e}\u{283}\u{292}\u{2a4}\u{2a7}\u{3b8}";
        const NON_QUOTE_PUNCTS: &str = "!,.:;?\u{2014}\u{2026}";

        // `next(..., vowel)` in Python: if no character belongs to any of the
        // three classes, the incoming value survives unchanged.
        let mut future_vowel = ctx.future_vowel;
        if let Some(ps) = ps {
            for c in ps.chars() {
                if NON_QUOTE_PUNCTS.contains(c) {
                    future_vowel = None;
                    break;
                } else if VOWELS.contains(c) {
                    future_vowel = Some(true);
                    break;
                } else if CONSONANTS.contains(c) {
                    future_vowel = Some(false);
                    break;
                }
            }
        }

        let future_to = tk.text == "to"
            || tk.text == "To"
            || (tk.text == "TO" && (tk.tag == "TO" || tk.tag == "IN"));

        crate::lexicon::TokenContext { future_vowel, future_to }
    }

    pub fn tokenize(&self, text: &str) -> Vec<MToken> {
        self.tokenize_with_offsets(text)
            .into_iter()
            .map(|(tk, _)| tk)
            .collect()
    }

    /// Tokenize, returning each subtoken alongside its byte range in `text`.
    ///
    /// Needed to map [`LinkMarker`] byte ranges (from
    /// [`G2P::preprocess_links`]) onto specific [`MToken`]s after
    /// subtokenization. The Python equivalent uses spaCy's
    /// `Alignment.from_strings` against a whitespace-split token list; we use
    /// byte offsets instead, which is more robust to subtoken splits.
    pub fn tokenize_with_offsets(
        &self,
        text: &str,
    ) -> Vec<(MToken, std::ops::Range<usize>)> {
        let word_boundary_regex = Regex::new(r"\S+").unwrap();
        let mut tokens = Vec::new();

        for mat in word_boundary_regex.find_iter(text) {
            let word = mat.as_str();
            let word_start = mat.start();
            let subtokens: Vec<regex::Match> =
                self.subtoken_regex.find_iter(word).collect();

            if subtokens.is_empty() {
                let tk = MToken::new(word.to_string(), "NN".to_string(), String::new());
                tokens.push((tk, word_start..mat.end()));
            } else {
                for sub in subtokens {
                    let s = word_start + sub.start();
                    let e = word_start + sub.end();
                    let tk =
                        MToken::new(sub.as_str().to_string(), "NN".to_string(), String::new());
                    tokens.push((tk, s..e));
                }
            }
        }

        // Fill in each token's trailing whitespace from the SOURCE TEXT.
        //
        // misaki carries spaCy's `token.whitespace_`, i.e. the actual
        // characters between this token and the next -- empty when the next
        // token is adjacent. The assembled phoneme string is
        // `''.join(phonemes + whitespace)` (`en.py:711`), so hardcoding " "
        // here produced "b\u02c8\u0251\u0279kt . " where misaki gives
        // "b\u02c8\u0251\u0279kt.", and no phoneme string could ever match.
        // Byte ranges from the regex are always char boundaries, so slicing is
        // safe.
        for i in 0..tokens.len() {
            let gap_start = tokens[i].1.end;
            let gap_end = if i + 1 < tokens.len() {
                tokens[i + 1].1.start
            } else {
                text.len()
            };
            if gap_start < gap_end {
                tokens[i].0.whitespace = text[gap_start..gap_end].to_string();
            }
        }

        tokens
    }

    pub fn g2p(&self, text: &str) -> Result<(String, Vec<MToken>), G2PError> {
        // Parse `[text](feature)` markers and attach them to the subtokens
        // whose byte range falls inside the marker span. Mirrors the
        // `Lexicon.tokenize` feature-application pass in misaki/en.py.
        let (processed_text, markers) = self.preprocess_links(text);
        let mut tokens_with_spans = self.tokenize_with_offsets(&processed_text);

        for marker in &markers {
            let mut first_in_span = true;
            for (tk, span) in tokens_with_spans.iter_mut() {
                if span.start >= marker.byte_start && span.end <= marker.byte_end {
                    match &marker.feature {
                        LinkFeature::Stress(s) => {
                            tk.underscore_mut().stress = Some(*s);
                        }
                        LinkFeature::Phonemes(p) => {
                            // Only the head subtoken receives the override;
                            // trailing subtokens are blanked so the main loop
                            // skips them. Matches Python's
                            // `phonemes = v.lstrip('/') if i == 0 else ''`.
                            if first_in_span {
                                tk.phonemes = Some(p.clone());
                            } else {
                                tk.phonemes = Some(String::new());
                            }
                        }
                        LinkFeature::NumFlags(flags) => {
                            tk.underscore_mut().num_flags = flags.clone();
                        }
                    }
                    first_in_span = false;
                }
            }
        }

        let mut tokens: Vec<MToken> =
            tokens_with_spans.into_iter().map(|(tk, _)| tk).collect();

        // Tag at WORD level, as misaki does. spaCy tags its own tokens and
        // misaki's subtokens inherit the parent's tag; our tokenizer splits
        // finer than spaCy's, so tagging subtokens asks the model about
        // fragments it never saw in training ("Mr" and "." separately rather
        // than "Mr."). Group by adjacency first -- the same grouping that
        // reassembles words later -- tag the group texts, and give every
        // subtoken its group's tag.
        //
        // norm is the lowercased text. spaCy also applies a norm-exceptions
        // table through its tokenizer ("n't" -> "not"), which only fires on
        // pieces its own tokenizer produces; ours does not produce them, and
        // the group text ("hadn't") is what gets tagged.
        // The tagger input must be spaCy's TOKENS, because misaki's grouping
        // rule is `tk.tag in PUNCT_TAGS` on those tokens -- and whether `_`
        // merges into its neighbour is decided by nothing else. spaCy tags the
        // `_` in `_like` as UH (misaki merges it) and the one in `_it` as NFP
        // (misaki does not), from structurally identical input.
        //
        // Our subtokens already agree with spaCy wherever it matters: `_`, `'`
        // and `’` come out separate, `It's` and `hadn't` come out whole, and
        // `five`/`-`/`storied` matches. The one systematic difference is the
        // period: we always split it, spaCy keeps its known abbreviations
        // together. So merge exactly those runs and tag the result.
        let groups_for_tagging = Self::group_for_tagging(&tokens);
        let tagger_input: Vec<(String, String, bool)> = groups_for_tagging
            .iter()
            .map(|g| {
                let sub = &tokens[g.clone()];
                let (t, _, ws) = Self::merge_text_tag(sub);
                // NORM is not lowercasing. spaCy resolves it through
                // BASE_NORMS, the tokenizer exceptions and lexeme_norm first,
                // and the entry that matters most here is the curly apostrophe
                // normalising to the straight one: with norm `’` the tagger
                // calls a possessive `''`, with norm `'` it calls it POS, and
                // misaki merges only the latter into its word.
                let norm = crate::norms::norm_of(&t, crate::spacy_tagger::hash_string);
                (t.clone(), norm, !ws.is_empty())
            })
            .collect();
        let group_tags = self.spacy_tagger.tag(&tagger_input);
        let mut tags: Vec<String> = Vec::with_capacity(tokens.len());
        for (gi, g) in groups_for_tagging.iter().enumerate() {
            let tag = group_tags.get(gi).copied().unwrap_or("NN").to_string();
            for _ in g.clone() {
                tags.push(tag.clone());
            }
        }

        log::debug!(
            "g2p '{}' -> {} tokens, {} tags",
            text,
            tokens.len(),
            tags.len()
        );
        for (i, tk) in tokens.iter().enumerate() {
            log::debug!("token[{}]: '{}'", i, tk.text);
        }

        // Process tokens in reverse order (like Python) to build context
        let mut contexts: Vec<crate::lexicon::TokenContext> =
            vec![crate::lexicon::TokenContext::default(); tokens.len()];

        // First, set tags
        for (tk, tag) in tokens.iter_mut().zip(tags.iter()) {
            tk.tag = tag.clone();
        }

        // misaki assigns punctuation its phonemes in `retokenize`, i.e. BEFORE
        // any lexicon lookup, and the resolution loop then skips those tokens
        // because `tk.phonemes is not None`. Order matters: our lookup
        // normalises ‘ to a straight quote, which the lexicon resolves to ”, so
        // reaching the lexicon at all gave ‘ the CLOSING glyph.
        //
        // The test is misaki's own, on the TAG. A character test was wrong in
        // both directions: it gave `[` an opening paren where spaCy tags it XX
        // rather than -LRB-, so misaki merges it into the following word and it
        // contributes NOTHING -- `[Illustration` is ˌɪləstrˈAʃən, not
        // `( ˌɪləstrˈAʃən`.
        for i in 0..tokens.len() {
            if tokens[i].phonemes.is_some() || tokens[i].text.is_empty() {
                continue;
            }
            if Self::is_punct_token(&tokens[i]) {
                tokens[i].phonemes = Some(Self::punct_phonemes(&tokens[i].text));
            }
        }

        // Process in reverse to build context from future tokens
        for i in (0..tokens.len()).rev() {
            // misaki normalises the LOOKUP KEY before touching the lexicon
            // (`Lexicon.__call__`):
            //
            //     word = tk.text.replace(chr(8216), "'").replace(chr(8217), "'")
            //     word = unicodedata.normalize('NFKC', word)
            //
            // The lexicons are keyed with STRAIGHT apostrophes -- "it's" is in
            // us_gold as ɪts, "I'm" as ˌIm, "'s" as s -- so prose using curly
            // ’ missed every one of them and fell through to spelling the
            // pieces out. The token's own text keeps the original character.
            let word = Self::normalize_lookup(&tokens[i].text);
            let tag = tokens[i].tag.clone();
            // An explicit `[word](+N)` marker takes precedence over
            // capitalization-derived stress, matching Python misaki which
            // writes `tk._.stress = v` before the lexicon lookup runs.
            let stress = tokens[i].underscore().stress.or_else(|| {
                if word == word.to_lowercase() {
                    None
                } else if word == word.to_uppercase() {
                    Some(self.lexicon.cap_stresses.1)
                } else {
                    Some(self.lexicon.cap_stresses.0)
                }
            });

            // NOTE: `contexts[i]` was already filled by iteration i+1 from that
            // token's RESOLVED PHONEMES -- see the `token_context` call at the
            // end of this loop. There was an orthographic heuristic here
            // ("does the next word start with aeiou") that overwrote it on
            // every iteration, which is why `the FBI` came out `\u00f0\u0259` instead of
            // `\u00f0i`: F is not in "aeiou", but FBI's phonemes begin `\u02cc\u025b`, a vowel.
            // misaki decides this from phonemes, never from spelling.

            // Process current token
            if tokens[i].phonemes.is_none() {
                let ctx = Some(&contexts[i]);

                // Use get_word which handles special cases, lookup, and stemming
                if let Some((ps, _)) = self.lexicon.get_word(&word, &tag, stress, ctx) {
                    tokens[i].phonemes = Some(ps);
                }

                if tokens[i].phonemes.is_none() {
                    if word.contains('-') && word.len() > 1 {
                        // Handle hyphenated words like "twenty-one"
                        let parts: Vec<&str> = word.split('-').filter(|s| !s.is_empty()).collect();
                        let mut sub_ps = Vec::new();
                        for part in parts {
                            let (p, _) = self.g2p(part)?;
                            sub_ps.push(p);
                        }
                        tokens[i].phonemes = Some(sub_ps.join(" "));
                    } else if self.is_number_like(&word) {
                        let spoken = self.convert_number(&word);
                        if spoken != word {
                            let (p, _) = self.g2p(&spoken)?;
                            tokens[i].phonemes = Some(p);
                        }
                    }
                }

                if tokens[i].phonemes.is_none() {
                    if let Some(ps) = self.rules.apply_rules(&word, &tag, &self.lexicon) {
                        tokens[i].phonemes = Some(ps);
                    }
                }

                if tokens[i].phonemes.is_none() {
                    if word.chars().count() > 1 {
                        // Unknown multi-character word - use fallback
                        let mut handled = false;
                        if let Some(ref fallback) = self.fallback {
                            match fallback.phonemize(&word) {
                                Ok(ps) => {
                                    tokens[i].phonemes = Some(ps);
                                    handled = true;
                                }
                                Err(e) => {
                                    log::error!("fallback error for '{}': {}", word, e);
                                    return Err(G2PError::Fallback(e));
                                }
                            }
                        }

                        if !handled {
                            // Leave it unresolved. misaki returns (None, None) for an
                            // out-of-vocabulary word (`Lexicon.__call__`); with no
                            // fallback configured the phonemes stay None and the
                            // assembled string substitutes `unk`.
                            //
                            // There was a character-by-character spell-out here. It is
                            // not what misaki does and it was actively harmful: every
                            // unknown proper noun became its spelling, so "Bennet" read
                            // as "b-e-n-n-e-t" and one token became six, which is why
                            // 257 of 413 failing prose cases differed in TOKEN COUNT
                            // and therefore in every downstream word timestamp.
                            let _ = &mut handled;
                        }
                    } else {
                        // Try to normalize the character or return unknown
                        let normalized: String = word
                            .chars()
                            .map(|c| match c {
                                'é' | 'è' | 'ê' | 'ë' => 'e',
                                'á' | 'à' | 'â' | 'ä' | 'ã' | 'å' => 'a',
                                'í' | 'ì' | 'î' | 'ï' => 'i',
                                'ó' | 'ò' | 'ô' | 'ö' | 'õ' => 'o',
                                'ú' | 'ù' | 'û' | 'ü' => 'u',
                                'ñ' => 'n',
                                'ç' => 'c',
                                // NB: em/en dashes are NOT mapped away here.
                                // U+2014 is in misaki's PUNCTS and survives as
                                // itself; mapping it to a space sent it down the
                                // renormalise path and silently dropped it.
                                _ => c,
                            })
                            .collect();

                        if normalized != word {
                            let (p, _) = self.g2p(&normalized)?;
                            tokens[i].phonemes = Some(p);
                        } else {
                            // Punctuation, as misaki resolves it (`en.py:598-599`):
                            //
                            //     elif tk.tag in PUNCT_TAGS and not all(97 <= ord(c.lower()) <= 122 ...):
                            //         tk.phonemes = PUNCT_TAG_PHONEMES.get(
                            //             tk.tag, ''.join(c for c in tk.text if c in PUNCTS))
                            //
                            // with PUNCTS = ';:,.!?\u2014\u2026"\u201c\u201d' and PUNCT_TAG_PHONEMES
                            // mapping spaCy's quote/paren tags: `` -> \u201c, '' -> \u201d,
                            // -LRB- -> (, -RRB- -> ).
                            //
                            // The punctuation pre-pass above already handled
                            // every token misaki calls punctuation, by TAG. A
                            // non-alphanumeric token that gets here is therefore
                            // one spaCy did NOT tag as punctuation -- `[` is
                            // tagged XX, not -LRB- -- and misaki leaves it
                            // unresolved, so it contributes nothing when merged
                            // into the following word. `[Illustration` is
                            // ˌɪləstrˈAʃən, not `( ˌɪləstrˈAʃən`.
                            //
                            // SUBTOKEN_JUNKS still resolve to the empty string
                            // (resolve_tokens), which the merge pass applies.
                            tokens[i].phonemes = None;
                        }
                    }
                }
            }

            // Propagate context leftwards, exactly as misaki's
            // `G2P.token_context` does (`en.py`):
            //
            //     vowel = next((None if c in NON_QUOTE_PUNCTS else (c in VOWELS)
            //                   for c in ps if any(c in s for s in
            //                   (VOWELS, CONSONANTS, NON_QUOTE_PUNCTS))), vowel) if ps else vowel
            //
            // Three things the previous version got wrong and this one does not:
            // punctuation RESETS future_vowel to None rather than being skipped;
            // a token whose phonemes contain no vowel, consonant or punctuation
            // CARRIES THE PREVIOUS value rather than leaving a stale one; and
            // future_to is set here from the token, not guessed from spelling.
            if i > 0 {
                contexts[i - 1] =
                    Self::token_context(&contexts[i], tokens[i].phonemes.as_deref(), &tokens[i]);
            }
        }

        // ---- merge adjacent subtokens into words (misaki's `merge_tokens`) ----
        //
        // misaki's final token list has ONE token per adjacency group:
        //
        //     tokens = [merge_tokens(tk, unk=self.unk) if isinstance(tk, list) else tk
        //               for tk in tokens]
        //
        // and this runs BEFORE the flap pass below, which is why it is placed
        // here. Without it our tokenizer's finer splits reached the output and
        // 257 of 700 prose cases differed in token count -- every one of which
        // corrupts the word timestamps, since `pred_dur` is per token.
        //
        // Phonemes are joined with no separator unless `prespace`, which misaki
        // derives in `resolve_tokens` from the group's text:
        //
        //     prespace = ' ' in text or '/' in text or len({0 if c.isalpha()
        //                else (1 if is_digit(c) else 2) for c in text
        //                if c not in SUBTOKEN_JUNKS}) > 1
        //
        // i.e. a space, a slash, or a MIX of character classes once the junk
        // characters are ignored. `five-storied` is all-alpha once `-` is
        // dropped, so it joins tight: fˌIv + stˈɔɹid -> fˌIvstˈɔɹid.
        {
            const SUBTOKEN_JUNKS: &str = "',-._\u{2018}\u{2019}/";
            let groups = Self::group_by_adjacency(&tokens);
            if groups.len() != tokens.len() {
                let mut merged: Vec<MToken> = Vec::with_capacity(groups.len());
                for g in groups {
                    if g.len() == 1 {
                        merged.push(tokens[g.start].clone());
                        continue;
                    }
                    let group = &tokens[g.clone()];
                    let (text, tag, ws) = Self::merge_text_tag(group);

                    // misaki's `resolve_tokens`: a subtoken made entirely of
                    // SUBTOKEN_JUNKS contributes NOTHING.
                    //
                    //     elif all(c in SUBTOKEN_JUNKS for c in tk.text):
                    //         tk.phonemes = ''
                    //
                    // Without this the word-internal ’ in `it’s` kept the ”
                    // that the standalone-quote path gives it, producing
                    // ɪt”ˈɛs instead of ɪts.
                    let part_of = |tk: &MToken| -> String {
                        if !tk.text.is_empty() && tk.text.chars().all(|c| SUBTOKEN_JUNKS.contains(c)) {
                            String::new()
                        } else {
                            tk.phonemes.clone().unwrap_or_default()
                        }
                    };

                    let classes: std::collections::BTreeSet<u8> = text
                        .chars()
                        .filter(|c| !SUBTOKEN_JUNKS.contains(*c))
                        .map(|c| if c.is_alphabetic() { 0u8 } else if c.is_ascii_digit() { 1 } else { 2 })
                        .collect();
                    let prespace =
                        text.contains(' ') || text.contains('/') || classes.len() > 1;

                    // misaki tries the whole word in the lexicon first and only
                    // falls back to stitching the parts together.
                    // misaki's multi-token loop (`G2P.__call__`). It does not
                    // pick one span -- it resolves the word in PIECES, right to
                    // left, and the result is their concatenation:
                    //
                    //     left, right = 0, len(w)
                    //     while left < right:
                    //         tk = merge_tokens(w[left:right])      # if none resolved yet
                    //         ps, rating = self.lexicon(tk, ctx)
                    //         if ps is not None:
                    //             w[left].phonemes = ps
                    //             for x in w[left+1:right]: x.phonemes = ''
                    //             right = left; left = 0
                    //         elif left + 1 < right: left += 1
                    //         else:
                    //             right -= 1
                    //             if all(c in SUBTOKEN_JUNKS for c in w[right].text):
                    //                 w[right].phonemes = ''
                    //             left = 0
                    //
                    // `Austen’s` needs this: "Austen's" misses, then the span
                    // "'s" HITS (us_gold holds "'s" as s), and Austen stays
                    // unresolved -- giving misaki's `s`. Trying only the whole
                    // word stitched on the letter name ˈɛs instead.
                    // misaki's `Lexicon.__call__` tries get_word, then the
                    // NUMBER path, before anything else:
                    //
                    //     ps, rating = self.get_word(...)
                    //     if ps is not None: return ...
                    //     elif Lexicon.is_number(word, tk._.is_head):
                    //         ps, rating = self.get_number(...)
                    //
                    // Our tokenizer splits `1st` into `1` + `st`, so the group
                    // text is the first point at which it looks like an ordinal
                    // at all. Without this `1st` read as "one" and `23rd` as
                    // "twenty three".
                    // Currency. misaki's `Lexicon.get_number` pairs the
                    // dot-separated parts with CURRENCIES[sym] -- $ is
                    // (dollar, cent), £ is (pound, pence), € is (euro, cent) --
                    // drops a zero part, joins the rest with "and", and uses the
                    // singular at exactly one:
                    //
                    //   $9        -> nine dollars
                    //   $1        -> one dollar
                    //   $0.50     -> fifty cents        (the zero major is dropped)
                    //   $1,250.75 -> one thousand two hundred fifty dollars
                    //                and seventy five cents
                    if let Some(spoken) = self.currency_words(&text) {
                        if let Ok((p, _)) = self.g2p(&spoken) {
                            // misaki keeps the SYMBOL as its own token with
                            // empty phonemes and hangs the words off the number:
                            // `$9` is [('$', ''), ('9', 'nˈIn dˈɑləɹz')]. Emitting
                            // one merged token instead gave the right audio with
                            // the wrong word boundaries.
                            let mut sym = group[0].clone();
                            let mut it = text.chars();
                            let s0 = it.next().unwrap();
                            sym.text = s0.to_string();
                            sym.tag = "$".to_string();
                            sym.whitespace = String::new();
                            sym.phonemes = Some(String::new());
                            merged.push(sym);

                            let mut tk = group[0].clone();
                            tk.text = it.collect();
                            tk.tag = tag;
                            tk.whitespace = ws;
                            tk.phonemes = Some(p.trim_end().to_string());
                            merged.push(tk);
                            continue;
                        }
                    }

                    if self.is_number_like(&text) {
                        let spoken = self.convert_number(&text);
                        if spoken != text {
                            if let Ok((p, _)) = self.g2p(&spoken) {
                                let mut tk = group[0].clone();
                                tk.text = text;
                                tk.tag = tag;
                                tk.whitespace = ws;
                                tk.phonemes = Some(p.trim_end().to_string());
                                merged.push(tk);
                                continue;
                            }
                        }
                    }

                    let n = g.len();
                    let mut res: Vec<Option<String>> = vec![None; n];
                    let mut left = 0usize;
                    let mut right = n;
                    while left < right {
                        // misaki only merges a span none of whose members has
                        // resolved yet.
                        let free = res[left..right].iter().all(|r| r.is_none());
                        let hit = if free {
                            let sub = &tokens[g.start + left..g.start + right];
                            let (t, tg, _) = Self::merge_text_tag(sub);
                            let k = Self::normalize_lookup(&t);
                            let st = if k == k.to_lowercase() {
                                None
                            } else if k == k.to_uppercase() {
                                Some(self.lexicon.cap_stresses.1)
                            } else {
                                Some(self.lexicon.cap_stresses.0)
                            };
                            self.lexicon
                                .get_word(&k, &tg, st, Some(&contexts[g.start]))
                                .map(|(ps, _)| ps)
                        } else {
                            None
                        };
                        if let Some(ps) = hit {
                            res[left] = Some(ps);
                            for k in left + 1..right {
                                res[k] = Some(String::new());
                            }
                            right = left;
                            left = 0;
                        } else if left + 1 < right {
                            left += 1;
                        } else {
                            right -= 1;
                            if res[right].is_none() {
                                let t = &tokens[g.start + right].text;
                                if !t.is_empty() && t.chars().all(|c| SUBTOKEN_JUNKS.contains(c)) {
                                    res[right] = Some(String::new());
                                }
                            }
                            left = 0;
                        }
                    }
                    // Anything still unresolved stays unresolved, as misaki
                    // leaves it: merge_tokens substitutes `unk` for None.
                    // The stitched path below assembles whatever `res` holds.
                    let whole: Option<String> = None;

                    let phonemes = whole.or_else(|| {
                        // misaki's `resolve_tokens` rebalances stress across the
                        // parts of one word before they are joined:
                        //
                        //     indices = [(PRIMARY_STRESS in tk.phonemes,
                        //                 stress_weight(tk.phonemes), i) ...]
                        //     if len(indices) == 2 and len(tokens[indices[0][2]].text) == 1:
                        //         demote indices[1]
                        //     elif len(indices) < 2 or sum(b for b,_,_ in indices) <= (len(indices)+1)//2:
                        //         return
                        //     for _,_,i in sorted(indices)[:len(indices)//2]: demote i
                        //
                        // where demote is `apply_stress(ps, -0.5)` -- primary
                        // becomes secondary. Without it a compound kept a primary
                        // stress on every part: `Caucus-Race` came out
                        // kˈɔkəsɹˈAs where misaki gives kˈɔkəsɹˌAs.
                        //
                        // `stress_weight` counts diphthongs double, which is what
                        // decides WHICH part is demoted: ɹˈAs weighs 5 (A is a
                        // diphthong) against kˈɔkəs's 6, so Race loses its primary.
                        let mut parts: Vec<String> = res
                            .iter()
                            .enumerate()
                            .map(|(i, r)| match r {
                                Some(ps) => ps.clone(),
                                // Unresolved: misaki writes `unk`, which our
                                // pipeline sets to the empty string. Fall back to
                                // whatever the per-token pass found only for a
                                // token the span loop never reached.
                                None => part_of(&group[i]),
                            })
                            .collect();
                        // `resolve_tokens` RETURNS before rebalancing when the
                        // parts are space-separated:
                        //
                        //     if prespace:
                        //         return
                        //
                        // so a mixed-class word keeps every primary stress.
                        // That is why misaki gives `50%` as fˈɪfti pəɹsˈɛnt and not
                        // fˌɪfti pəɹsˈɛnt -- '5','0' and '%' are different classes,
                        // prespace is true, and no demotion happens.
                        if !prespace {
                            const PRIMARY: char = '\u{2c8}';
                            const DIPHTHONGS: &str = "AIOQWY\u{2a4}\u{2a7}";
                            let weight = |ps: &str| -> usize {
                                ps.chars()
                                    .map(|c| if DIPHTHONGS.contains(c) { 2 } else { 1 })
                                    .sum()
                            };
                            let mut idx: Vec<(bool, usize, usize)> = parts
                                .iter()
                                .enumerate()
                                .filter(|(_, p)| !p.is_empty())
                                .map(|(i, p)| (p.contains(PRIMARY), weight(p), i))
                                .collect();
                            let n_primary = idx.iter().filter(|(b, _, _)| *b).count();
                            if idx.len() == 2
                                && group[idx[0].2].text.chars().count() == 1
                            {
                                let i = idx[1].2;
                                parts[i] = self.lexicon.apply_stress(&parts[i], Some(-0.5));
                            } else if idx.len() >= 2 && n_primary > (idx.len() + 1) / 2 {
                                idx.sort();
                                let half = idx.len() / 2;
                                for (_, _, i) in idx.into_iter().take(half) {
                                    parts[i] = self.lexicon.apply_stress(&parts[i], Some(-0.5));
                                }
                            }
                        }

                        let mut out = String::new();
                        for part in parts {
                            if prespace
                                && !out.is_empty()
                                && !out.ends_with(char::is_whitespace)
                                && !part.is_empty()
                            {
                                out.push(' ');
                            }
                            out.push_str(&part);
                        }
                        Some(out)
                    });

                    let mut tk = group[0].clone();
                    tk.text = text;
                    tk.tag = tag;
                    tk.whitespace = ws;
                    tk.phonemes = phonemes;
                    merged.push(tk);
                }
                tokens = merged;
            }
        }

        // misaki applies exactly this, after all resolution and merging and
        // before assembling the string (`en.py:708-710`, gated on
        // `version != '2.0'`):
        //
        //     tk.phonemes = tk.phonemes.replace('\u027e', 'T').replace('\u0294', 't')
        //
        // The lexicons store \u027e and \u0294; Kokoro's vocab wants T and t. It runs
        // AFTER resolution on purpose -- the context pass matches \u027e against its
        // consonant set, so replacing earlier would break that -- and it is
        // applied to the TOKENS, not just the assembled string, because
        // `pred_dur` has one entry per token and the word timestamps are
        // derived from the per-token phonemes.
        for tk in tokens.iter_mut() {
            if let Some(ps) = tk.phonemes.as_mut() {
                if !ps.is_empty() {
                    *ps = ps.replace('\u{27e}', "T").replace('\u{294}', "t");
                }
            }
        }

        let result = tokens
            .iter()
            .map(|tk| tk.phonemes.as_ref().unwrap_or(&self.unk).clone() + &tk.whitespace)
            .collect::<String>();

        Ok((result, tokens))
    }

    /// Does this token go down misaki's number path?
    ///
    /// misaki's `Lexicon.is_number` peels an alphabetic suffix first, so `1st`,
    /// `23rd` and `2nd` are numbers, not words. Testing only for all-digits
    /// meant an ordinal never reached `get_number` and `1st` read as "one".
    fn is_number_like(&self, word: &str) -> bool {
        let clean = word.replace(',', "");
        let lower = clean.to_lowercase();
        for suf in ["st", "nd", "rd", "th"] {
            if let Some(d) = lower.strip_suffix(suf) {
                if !d.is_empty() && d.chars().all(|c| c.is_ascii_digit()) {
                    return true;
                }
            }
        }
        if let Some(rest) = clean.strip_prefix('-') {
            if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()) {
                return true;
            }
        }
        self.is_number(word)
    }

    fn is_number(&self, word: &str) -> bool {
        let clean = word.replace(",", "");
        clean.parse::<i64>().is_ok()
    }

    /// Reduce a num2words string to the word sequence misaki feeds the lexicon.
    ///
    /// misaki does `re.split(r'[^a-z]+', num2words(int(num)))` and looks each
    /// piece up on its own (`Lexicon.get_number`), so hyphens and commas become
    /// word boundaries -- "fifty-seven" is two lookups, not one. It also drops
    /// "and" unless '&' is in num_flags, which is why 2001 reads
    /// "two thousand one".
    ///
    /// Without this, "457" produced `f\u02c8\u0254\u0279 h\u02c8\u028cndr\u0259d f\u02c8\u026afti s\u02c8\u025bv\u0259n` with the last
    /// two words run together, because "fifty-seven" went to the lexicon whole.
    fn normalize_num_words(spoken: &str) -> String {
        spoken
            .split(|c: char| !c.is_ascii_alphabetic())
            .filter(|w| !w.is_empty() && *w != "and")
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// num2words' `to='year'` form, which misaki uses for any bare 4-digit
    /// number that is not a currency amount (`en.py`:
    /// `len(word) == 4 and currency not in CURRENCIES`).
    ///
    /// Verified against Python num2words: 1999 "nineteen ninety-nine",
    /// 1900 "nineteen hundred", 2000 "two thousand", 2001 "two thousand and
    /// one", 2005 "two thousand and five", 2010 "twenty ten", 1066 "ten
    /// sixty-six", 1905 "nineteen oh-five", 1100 "eleven hundred", 2100
    /// "twenty-one hundred", 1010 "ten ten".
    fn year_words(y: i64) -> Option<String> {
        let card = |n: i64| Num2Words::new(n).to_words().ok();
        let (hi, lo) = (y / 100, y % 100);
        if lo == 0 {
            if y % 1000 == 0 {
                card(y) // 2000 -> two thousand
            } else {
                Some(format!("{} hundred", card(hi)?)) // 1900 -> nineteen hundred
            }
        } else if hi % 10 == 0 && lo < 10 {
            card(y) // 2005 -> two thousand and five
        } else if lo < 10 {
            Some(format!("{} oh {}", card(hi)?, card(lo)?)) // 1905 -> nineteen oh-five
        } else {
            Some(format!("{} {}", card(hi)?, card(lo)?)) // 1999 -> nineteen ninety-nine
        }
    }

    /// A currency amount spelled out, or None if `word` is not one.
    ///
    /// See the call site for misaki's rules; the units come from its CURRENCIES
    /// table.
    fn currency_words(&self, word: &str) -> Option<String> {
        let mut cs = word.chars();
        let sym = cs.next()?;
        let units: (&str, &str) = match sym {
            '$' => ("dollar", "cent"),
            '\u{a3}' => ("pound", "pence"),
            '\u{20ac}' => ("euro", "cent"),
            _ => return None,
        };
        let rest: String = cs.collect::<String>().replace(',', "");
        if rest.is_empty() || !rest.chars().all(|c| c.is_ascii_digit() || c == '.') {
            return None;
        }
        let mut parts: Vec<(i64, &str)> = Vec::new();
        for (i, seg) in rest.split('.').enumerate().take(2) {
            let n: i64 = if seg.is_empty() { 0 } else { seg.parse().ok()? };
            parts.push((n, if i == 0 { units.0 } else { units.1 }));
        }
        // misaki drops a zero part rather than saying "zero dollars".
        if parts.len() > 1 {
            if parts[1].0 == 0 {
                parts.truncate(1);
            } else if parts[0].0 == 0 {
                parts.remove(0);
            }
        }
        let mut out: Vec<String> = Vec::new();
        for (n, unit) in parts {
            if !out.is_empty() {
                out.push("and".to_string());
            }
            out.push(Self::normalize_num_words(
                &Num2Words::new(n).to_words().ok()?,
            ));
            out.push(if n == 1 {
                unit.to_string()
            } else {
                format!("{unit}s")
            });
        }
        Some(out.join(" "))
    }

    fn convert_number(&self, word: &str) -> String {
        let clean = word.replace(",", "");

        // misaki prefixes "minus" for a leading '-' (`Lexicon.get_number`):
        //
        //     if word.startswith('-'):
        //         result.append(self.lookup('minus', ...))
        //         word = word[1:]
        //
        // We were dropping the sign, so -40 read as "forty".
        if let Some(rest) = clean.strip_prefix('-') {
            if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit() || c == '.') {
                return format!("minus {}", self.convert_number(rest));
            }
        }

        // Ordinals. misaki (`Lexicon.get_number`) peels an alphabetic suffix
        // and, when it is one of ORDINALS = {st, nd, rd, th} and the rest is
        // digits, reads the ORDINAL form:
        //
        //     if is_digit(word) and suffix in ORDINALS:
        //         extend_num(num2words(int(word), to='ordinal'), escape=True)
        //
        // Without it "23rd" read as "twenty three" and "1st" as "one".
        {
            let lower = clean.to_lowercase();
            for suf in ["st", "nd", "rd", "th"] {
                if let Some(digits) = lower.strip_suffix(suf) {
                    if !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()) {
                        if let Ok(n) = digits.parse::<i64>() {
                            if let Ok(spoken) = Num2Words::new(n).ordinal().to_words() {
                                return Self::normalize_num_words(&spoken);
                            }
                        }
                    }
                }
            }
        }

        // Bare four-digit numbers read as years, per misaki's get_number.
        if clean.len() == 4 && clean.chars().all(|c| c.is_ascii_digit()) {
            if let Ok(y) = clean.parse::<i64>() {
                if let Some(spoken) = Self::year_words(y) {
                    return Self::normalize_num_words(&spoken);
                }
            }
        }

        if let Ok(val) = clean.parse::<i64>() {
            let n2w = match self.lexicon.lang {
                Language::EnglishUS | Language::EnglishGB => Num2Words::new(val),
                // Language::Italian => Num2Words::new(val).lang(num2words::Lang::English),
            };
            if let Ok(spoken) = n2w.to_words() {
                return Self::normalize_num_words(&spoken);
            }
        }
        word.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestFallback;

    impl Fallback for TestFallback {
        fn phonemize(&self, word: &str) -> Result<String, FallbackError> {
            assert_eq!(word, "qzxwv");
            Ok("test-phonemes".to_string())
        }
    }

    #[test]
    fn test_g2p_basic() {
        let _ = env_logger::try_init();
        let g2p = G2P::new(Language::EnglishUS);
        let (phonemes, _) = g2p.g2p("Hello, world!").unwrap();
        println!("Phonemes: {}", phonemes);
        assert!(!phonemes.contains("❓"));
    }

    #[test]
    fn test_with_fallback_uses_injected_fallback_for_oov_word() {
        let g2p = G2P::with_fallback(Language::EnglishUS, Some(Box::new(TestFallback)));

        let (phonemes, tokens) = g2p.g2p("qzxwv").unwrap();

        assert_eq!(phonemes.trim(), "test-phonemes");
        assert_eq!(tokens[0].phonemes.as_deref(), Some("test-phonemes"));
    }

    // Tests for `[text](feature)` markdown-link parsing — see
    // misaki/en.py::Lexicon.preprocess for the reference implementation.

    #[test]
    fn test_preprocess_links_strips_markup() {
        let g2p = G2P::new(Language::EnglishUS);
        let (stripped, markers) = g2p.preprocess_links("say [hello](+2) please");
        assert_eq!(stripped, "say hello please");
        assert_eq!(markers.len(), 1);
        let m = &markers[0];
        assert_eq!(&stripped[m.byte_start..m.byte_end], "hello");
        match m.feature {
            LinkFeature::Stress(n) => assert!((n - 2.0).abs() < f64::EPSILON),
            _ => panic!("expected Stress, got {:?}", m.feature),
        }
    }

    #[test]
    fn test_preprocess_links_parses_features() {
        // Integer stress — signed and unsigned.
        assert!(matches!(parse_link_feature("+2"), Some(LinkFeature::Stress(n)) if n == 2.0));
        assert!(matches!(parse_link_feature("-1"), Some(LinkFeature::Stress(n)) if n == -1.0));
        assert!(matches!(parse_link_feature("2"),  Some(LinkFeature::Stress(n)) if n == 2.0));
        // Half stress — only these three literals are accepted upstream.
        assert!(matches!(parse_link_feature("0.5"),  Some(LinkFeature::Stress(n)) if n == 0.5));
        assert!(matches!(parse_link_feature("+0.5"), Some(LinkFeature::Stress(n)) if n == 0.5));
        assert!(matches!(parse_link_feature("-0.5"), Some(LinkFeature::Stress(n)) if n == -0.5));
        // Phoneme override.
        assert!(matches!(parse_link_feature("/həˈloʊ/"), Some(LinkFeature::Phonemes(ref p)) if p == "həˈloʊ"));
        // Num flags.
        assert!(matches!(parse_link_feature("#cardinal#"), Some(LinkFeature::NumFlags(ref f)) if f == "cardinal"));
        // Anything else → None, matching the upstream `else: f = None` branch.
        assert!(parse_link_feature("").is_none());
        assert!(parse_link_feature("=").is_none());
        assert!(parse_link_feature("doctor").is_none());
        assert!(parse_link_feature("2.5").is_none());
        assert!(parse_link_feature("/foo").is_none());
        assert!(parse_link_feature("foo/").is_none());
    }

    #[test]
    fn test_link_stress_promotes() {
        // `[hello](+2)` should phonemize successfully and emit primary stress.
        let g2p = G2P::new(Language::EnglishUS);
        let (boosted, _) = g2p.g2p("[hello](+2)").unwrap();
        assert!(!boosted.contains("❓"), "boosted output: '{}'", boosted);
        assert!(boosted.contains('ˈ'),
            "boosted output missing primary stress: '{}'", boosted);
    }

    #[test]
    fn test_link_stress_strips() {
        // `(-2)` strips all stress marks via Lexicon::apply_stress.
        let g2p = G2P::new(Language::EnglishUS);
        let (stripped, _) = g2p.g2p("[hello](-2)").unwrap();
        assert!(!stripped.contains("❓"));
        assert!(!stripped.contains('ˈ') && !stripped.contains('ˌ'),
            "expected no stress marks, got: '{}'", stripped);
    }

    #[test]
    fn test_link_phoneme_override() {
        // `/.../` content is emitted verbatim, bypassing the lexicon.
        let g2p = G2P::new(Language::EnglishUS);
        let (out, _) = g2p.g2p("[xyzzy](/həˈloʊ/)").unwrap();
        assert!(out.contains("həˈloʊ"), "phoneme override missing: '{}'", out);
        assert!(!out.contains("❓"), "got unknown marker: '{}'", out);
    }

    #[test]
    fn test_link_num_flags_sets_field() {
        // `#flag#` markers don't affect phonemes today (the number-spelling
        // logic that reads `num_flags` isn't ported yet), but the field must
        // be set on the token so a future port can consume it.
        let g2p = G2P::new(Language::EnglishUS);
        let (_, tokens) = g2p.g2p("[5](#cardinal#)").unwrap();
        let tagged: Vec<_> = tokens
            .iter()
            .filter(|t| !t.underscore().num_flags.is_empty())
            .collect();
        assert_eq!(tagged.len(), 1, "expected one token with num_flags set");
        assert_eq!(tagged[0].underscore().num_flags, "cardinal");
    }

    #[test]
    fn test_link_unknown_feature_is_dropped() {
        // `=` and bare words must not parse — they fall through to None and
        // the marker is silently ignored, matching upstream behaviour.
        let g2p = G2P::new(Language::EnglishUS);
        let (stripped, markers) = g2p.preprocess_links("say [hello](=) please");
        assert_eq!(stripped, "say hello please");
        assert!(markers.is_empty(), "unknown feature should yield no marker");
    }

    /// With espeak feature: "eBook" is phonemized by espeak fallback (not in lexicon).
    #[test]
    #[cfg(feature = "espeak")]
    fn test_ebook_with_espeak() {
        let g2p = G2P::new(Language::EnglishUS);
        let (phonemes, _) = g2p.g2p("eBook").unwrap();
        assert!(
            !phonemes.contains("❓"),
            "with espeak: 'eBook' should be phonemized by fallback, got: {}",
            phonemes
        );
        // Should not be spelled out letter-by-letter (e.g. no ˈi for 'e' as letter)
        assert!(
            !phonemes.contains("ˈɛl"),
            "with espeak: 'eBook' should not spell out letters, got: {}",
            phonemes
        );
        println!("eBook (with espeak): {}", phonemes);
    }

    /// Without espeak feature: "eBook" is OOV so we get unknown marker or character spelling.
    #[test]
    #[cfg(not(feature = "espeak"))]
    fn test_ebook_without_espeak() {
        let g2p = G2P::new(Language::EnglishUS);
        let (phonemes, _) = g2p.g2p("eBook").unwrap();
        // No fallback: either ❓ for unknown or character-by-character spelling
        assert!(
            phonemes.contains("❓") || phonemes.contains("ˈi") || phonemes.contains("b"),
            "without espeak: 'eBook' should show unknown or spelled form, got: {}",
            phonemes
        );
        println!("eBook (without espeak): {}", phonemes);
    }

    // #[test]
    // fn test_g2p_italian() {
    //     let g2p = G2P::new(Language::Italian);
    //     let (phonemes, _) = g2p.g2p("Ciao, mondo!");
    //     println!("Phonemes: {}", phonemes);
    //     // "ciao" -> c+i+a+o -> tʃ+a+o -> with stress tʃˈao
    //     // "mondo" -> m+o+n+d+o -> mˈondo
    //     assert!(phonemes.contains("tʃ") && phonemes.contains("ao"));
    //     assert!(phonemes.contains("mondo"));
    // }

    // #[test]
    // fn test_convert_number_italian() {
    //     let g2p = G2P::new(Language::Italian);
    //     let (phonemes, _) = g2p.g2p("42");
    //     println!("Phonemes for 42: {}", phonemes);
    //     // 42 in Italian is "quarantadue" -> kwarantadue
    //     // We relax the check to ensure it produces phonemes and not numbers/unknowns
    //     assert!(!phonemes.contains("42"));
    //     assert!(!phonemes.contains("❓"));
    //     assert!(phonemes.contains("kwaranta") || phonemes.contains("due"));
    // }

    #[test]
    fn test_english_abbreviations() {
        let g2p = G2P::new(Language::EnglishUS);
        let cases = vec![
            "I'll",
            "I've",
            "it's",
            "he's",
            "she's",
            "we're",
            "they're",
            "isn't",
            "aren't",
            "wasn't",
            "weren't",
            "don't",
            "doesn't",
            "didn't",
            "can't",
            "couldn't",
            "shouldn't",
            "wouldn't",
            "won't",
            "hasn't",
            "haven't",
            "hadn't",
            "let's",
            "that's",
            "what's",
            "who's",
            "here's",
            "there's",
            "where's",
            "how's",
        ];
        for text in cases {
            let (p, _) = g2p.g2p(text).unwrap();
            println!("'{}' -> '{}'", text, p);
            assert!(!p.contains("❓"), "Failed for '{}'", text);
        }
    }

    #[test]
    fn test_casing_and_special_chars() {
        let g2p = G2P::new(Language::EnglishUS);

        // Test 1: All caps with suffix
        let (playing, _) = g2p.g2p("PLAYING").unwrap();
        println!("PLAYING: {}", playing);
        assert!(
            !playing.contains("❓"),
            "PLAYING should be resolved, got: {}",
            playing
        );

        // Test 2: Contractions
        let (ive, _) = g2p.g2p("I've").unwrap();
        println!("I've: {}", ive);
        assert!(!ive.contains("❓"), "I've should be resolved, got: {}", ive);

        // Test 3: Dashes
        // em-dash — (U+2014) and hyphen -
        let (dash, _) = g2p.g2p("word - word — word").unwrap();
        println!("Dash: {}", dash);
        // misaki 0.9.4 gives exactly this, verified against the oracle:
        //
        //     "word - word — word" -> wˈɜɹd ❓ wˈɜɹd — wˈɜɹd
        //
        // The two dashes are NOT alike. spaCy tags the em-dash `:`, which puts
        // it in PUNCT_TAGS and renders it —; it tags the plain hyphen HYPH,
        // which is not punctuation, so the lexicon misses it and it comes back
        // unresolved and prints as `unk`. This assertion used to forbid `unk`
        // altogether, which was the pre-fork behaviour.
        assert!(dash.contains('—'), "em-dash dropped: {dash}");
        assert_eq!(
            dash, "wˈɜɹd ❓ wˈɜɹd — wˈɜɹd",
            "does not match misaki 0.9.4"
        );
    }

    #[test]
    fn test_kokoros_basic() {
        let g2p = G2P::new(Language::EnglishUS);
        let cases = vec![
            "hello",
            "world",
            "the quick brown fox",
            "testing phonemization",
            "Hello, world!",
            "123",
            "restriction",
            "restrictions",
            "",
        ];
        for text in cases {
            let (p, _) = g2p.g2p(text).unwrap();
            println!("'{}' -> '{}'", text, p);
            if !text.is_empty() {
                assert!(!p.is_empty(), "Failed for '{}'", text);
            }
        }
    }

    #[test]
    fn test_kokoros_numbers() {
        let g2p = G2P::new(Language::EnglishUS);
        let cases = vec![
            "CHAPTER XIV",
            "CHAPTER 14",
            "CHAPTER 123",
            "I have 5 apples and 42 oranges",
            "The year 2024",
            "1234567890",
            "CHAPTER I",
            "CHAPTER II",
            "CHAPTER III",
            "CHAPTER IV",
            "CHAPTER V",
            "CHAPTER X",
            "CHAPTER XX",
            "CHAPTER XXX",
            "In 2024, CHAPTER XIV had 42 pages.",
            "The price is $123.45",
            "Temperature: -5°C",
            "Score: 100%",
            "Version 2.0",
            "3.14159",
        ];
        for text in cases {
            let (p, _) = g2p.g2p(text).unwrap();
            println!("'{}' -> '{}'", text, p);
            assert!(!p.is_empty(), "Failed for '{}'", text);
        }
    }

    #[test]
    fn test_kokoros_utf8_and_special() {
        let g2p = G2P::new(Language::EnglishUS);
        let cases = vec![
            "café",
            "naïve",
            "résumé",
            "Zürich",
            "São Paulo",
            "Müller",
            "北京",
            "こんにちは",
            "Здравствуй",
            "مرحبا",
            "🎉🎊🎈",
            // Control chars
            "\x00\x01\x02",
            // Mixed scripts
            "Hello 世界",
            "123中文",
            "English123中文",
            // Zero-width characters
            "hello\u{200B}world", // zero-width space
            "hello\u{200C}world", // zero-width non-joiner
            "hello\u{200D}world", // zero-width joiner
            // Combining characters
            "caf\u{00E9}",  // é as combining character
            "na\u{00EF}ve", // ï as combining character
        ];
        for text in cases {
            let (p, _) = g2p.g2p(text).unwrap();
            println!("'{}' -> '{}'", text, p);
            // Some might be empty/unknown depending on handling, but shouldn't crash
        }
    }

    #[test]
    fn test_kokoros_punctuation() {
        let g2p = G2P::new(Language::EnglishUS);
        let cases = vec![
            "Hello—world", // em dash
            "Hello–world", // en dash
            "Hello…world", // ellipsis
            "\"quoted text\"",
            "'single quotes'",
            "«French quotes»",
            "„German quotes„",
            "「Japanese quotes」",
            "Dr. Smith",
            "Mr. Jones",
            "Mrs. Brown",
            "Ms. Davis",
            "etc.",
            "U.S.A.",
            "Ph.D.",
            "A.I.",
            "NASA",
            "FBI",
            "   ",
            "\n\n",
            "\t\t",
            "\r\n",
        ];
        for text in cases {
            let (p, _) = g2p.g2p(text).unwrap();
            println!("'{}' -> '{}'", text, p);
        }
    }

    #[test]
    fn test_kokoros_punctuation_preserved() {
        // Kokoro's vocab encodes ";:,.!?" as distinct token ids (1-6).
        // G2P must emit each character literally in the phoneme stream so
        // downstream tokenizers can map them; collapsing to space loses the
        // sentence-end / clause-breath / question-intonation cues the
        // synthesis model relies on. Other ASCII punctuation (parens,
        // dashes, ellipsis, …) keeps the prior single-space fallback.
        let g2p = G2P::new(Language::EnglishUS);

        let (phonemes, _) = g2p.g2p("Hello, world.").unwrap();
        assert!(phonemes.contains(','), "comma dropped: {phonemes:?}");
        assert!(phonemes.contains('.'), "period dropped: {phonemes:?}");

        for c in [';', ':', '!', '?'] {
            let input = format!("test {c}");
            let (out, _) = g2p.g2p(&input).unwrap();
            assert!(out.contains(c), "{c:?} dropped: {out:?}");
        }

        // Parentheses ARE preserved. This assertion used to require the
        // opposite, which was the pre-fork behaviour: misaki maps spaCy's
        // -LRB-/-RRB- tags to '(' and ')' via PUNCT_TAG_PHONEMES
        // (`en.py:598-599`), and misaki 0.9.4 gives
        //
        //     "hello (world)" -> həlˈO (wˈɜɹld)
        //
        // verified against the oracle. The gate is misaki's output, so the test
        // follows it rather than the other way round.
        let (out, _) = g2p.g2p("hello (world)").unwrap();
        assert!(out.contains('('), "( dropped: {out:?}");
        assert!(out.contains(')'), ") dropped: {out:?}");
    }

    #[test]
    fn test_kokoros_long_text() {
        let g2p = G2P::new(Language::EnglishUS);
        // Reduced to 100 to check if it crashes
        let long_text = "a".repeat(1000);
        let (p, _) = g2p.g2p(&long_text).unwrap();
        assert!(!p.is_empty());
    }
}
