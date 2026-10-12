//! spaCy's English tokenizer, ported so misaki's word grouping can read the
//! tag of a spaCy token.
//!
//! misaki tokenizes with spaCy and subtokenizes within each spaCy token, and
//! its grouping rule is `tk.tag in PUNCT_TAGS` on those tokens. Everything else
//! in this crate now matches; what remained were punctuation runs where spaCy's
//! prefix/suffix peeling decides whether a mark is its own token:
//!
//!     "17—."            one token        "17."           17 + .
//!     "loathsome!--and"  one token        "else—but"      else + — + but
//!     "can;—but"         one token        "five-storied"  five + - + storied
//!
//! The difference between the first two is a lookbehind in spaCy's suffix
//! pattern: the `.` rule only fires after a character in a large class that
//! includes digits and letters but not the em dash. Porting the patterns
//! verbatim (`src/spacy_tokenizer_data.rs`, taken from the live tokenizer's
//! compiled searchers) gets all of this for free; reverse-engineering the
//! behaviour case by case does not.
//!
//! The algorithm is `Tokenizer._split_affixes` and `._attach_tokens`
//! (`spacy/tokenizer.pyx`), transcribed.

use std::collections::HashMap;

use fancy_regex::Regex;

use crate::spacy_tokenizer_data as data;

pub struct SpacyTokenizer {
    prefix: Regex,
    suffix: Regex,
    infix: Regex,
    url: Option<Regex>,
    specials: HashMap<&'static str, &'static [&'static str]>,
}

impl Default for SpacyTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl SpacyTokenizer {
    pub fn new() -> Self {
        SpacyTokenizer {
            prefix: Regex::new(data::PREFIX_PATTERN).expect("prefix pattern"),
            suffix: Regex::new(data::SUFFIX_PATTERN).expect("suffix pattern"),
            infix: Regex::new(data::INFIX_PATTERN).expect("infix pattern"),
            url: Regex::new(data::URL_PATTERN).ok(),
            specials: data::EXCEPTIONS.iter().copied().collect(),
        }
    }

    /// Byte length of the prefix to peel, or 0.
    fn find_prefix(&self, s: &str) -> usize {
        match self.prefix.find(s) {
            Ok(Some(m)) => m.end() - m.start(),
            _ => 0,
        }
    }

    /// Byte length of the suffix to peel, or 0.
    fn find_suffix(&self, s: &str) -> usize {
        match self.suffix.find(s) {
            Ok(Some(m)) => m.end() - m.start(),
            _ => 0,
        }
    }

    fn is_special(&self, s: &str) -> bool {
        self.specials.contains_key(s)
    }

    fn is_url(&self, s: &str) -> bool {
        self.url
            .as_ref()
            .map(|r| matches!(r.is_match(s), Ok(true)))
            .unwrap_or(false)
    }

    /// `Tokenizer._split_affixes`: peel prefixes and suffixes alternately,
    /// stopping as soon as what is left is a special case, and return the
    /// middle. Prefixes come back in order; suffixes come back OUTERMOST FIRST
    /// and are emitted reversed, as spaCy does with its reverse iterator.
    fn split_affixes<'a>(
        &self,
        mut s: &'a str,
        prefixes: &mut Vec<&'a str>,
        suffixes: &mut Vec<&'a str>,
    ) -> &'a str {
        let mut last_size = usize::MAX;
        while !s.is_empty() && s.len() != last_size {
            if self.is_special(s) {
                break;
            }
            last_size = s.len();

            let pre_len = self.find_prefix(s);
            let mut minus_pre = "";
            if pre_len != 0 {
                minus_pre = &s[pre_len..];
                if !minus_pre.is_empty() && self.is_special(minus_pre) {
                    prefixes.push(&s[..pre_len]);
                    s = minus_pre;
                    break;
                }
            }

            // spaCy searches the suffix in the string AFTER the prefix, but
            // slices it off the FULL string; identical, since it is at the end.
            let suf_len = self.find_suffix(&s[pre_len..]);
            let mut minus_suf = "";
            if suf_len != 0 {
                minus_suf = &s[..s.len() - suf_len];
                if !minus_suf.is_empty() && self.is_special(minus_suf) {
                    suffixes.push(&s[s.len() - suf_len..]);
                    s = minus_suf;
                    break;
                }
            }

            if pre_len != 0 && suf_len != 0 && pre_len + suf_len <= s.len() {
                prefixes.push(&s[..pre_len]);
                suffixes.push(&s[s.len() - suf_len..]);
                s = &s[pre_len..s.len() - suf_len];
            } else if pre_len != 0 {
                prefixes.push(&s[..pre_len]);
                s = minus_pre;
            } else if suf_len != 0 {
                suffixes.push(&s[s.len() - suf_len..]);
                s = minus_suf;
            }
        }
        s
    }

    /// `Tokenizer._attach_tokens`: prefixes, then the middle (expanded as a
    /// special case, kept whole as a URL, or split on infixes), then the
    /// suffixes in reverse.
    fn attach_tokens(&self, mid: &str, prefixes: &[&str], suffixes: &[&str], out: &mut Vec<String>) {
        for p in prefixes {
            out.push((*p).to_string());
        }
        if !mid.is_empty() {
            if let Some(pieces) = self.specials.get(mid) {
                for p in *pieces {
                    out.push((*p).to_string());
                }
            } else if self.is_url(mid) {
                out.push(mid.to_string());
            } else {
                let mut start = 0usize;
                let mut any = false;
                let mut it = self.infix.find_iter(mid);
                let mut matches: Vec<(usize, usize)> = Vec::new();
                while let Some(Ok(m)) = it.next() {
                    matches.push((m.start(), m.end()));
                }
                for (ms, me) in matches {
                    // spaCy skips a match flush against the start, so a split
                    // never produces an empty leading token.
                    if ms == 0 {
                        continue;
                    }
                    any = true;
                    if ms != start {
                        out.push(mid[start..ms].to_string());
                    }
                    if ms != me {
                        out.push(mid[ms..me].to_string());
                    }
                    start = me;
                }
                if !any {
                    out.push(mid.to_string());
                } else if start < mid.len() {
                    out.push(mid[start..].to_string());
                }
            }
        }
        for s in suffixes.iter().rev() {
            out.push((*s).to_string());
        }
    }

    /// Tokenize one whitespace-delimited chunk.
    fn tokenize_chunk(&self, chunk: &str, out: &mut Vec<String>) {
        if let Some(pieces) = self.specials.get(chunk) {
            for p in *pieces {
                out.push((*p).to_string());
            }
            return;
        }
        let mut prefixes: Vec<&str> = Vec::new();
        let mut suffixes: Vec<&str> = Vec::new();
        let mid = self.split_affixes(chunk, &mut prefixes, &mut suffixes);
        self.attach_tokens(mid, &prefixes, &suffixes, out);
    }

    /// Tokenize `text`, returning `(token, trailing whitespace)` pairs, the
    /// same shape spaCy's `token.text` / `token.whitespace_` give misaki.
    pub fn tokenize(&self, text: &str) -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = Vec::new();
        let mut i = 0usize;
        let bytes = text.as_bytes();
        while i < text.len() {
            // skip leading whitespace (it attaches to the PREVIOUS token)
            if bytes[i].is_ascii_whitespace() {
                let start = i;
                while i < text.len() && bytes[i].is_ascii_whitespace() {
                    i += 1;
                }
                if let Some(last) = out.last_mut() {
                    last.1.push_str(&text[start..i]);
                }
                continue;
            }
            let start = i;
            while i < text.len() && !bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            let mut pieces: Vec<String> = Vec::new();
            self.tokenize_chunk(&text[start..i], &mut pieces);
            for p in pieces {
                out.push((p, String::new()));
            }
        }
        out
    }
}
