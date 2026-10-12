//! How far is our POS tagger from spaCy's, and how much of the parity gap is
//! attributable to it? Compares tags only on cases where TOKENIZATION already
//! agrees, so tokenizer differences do not pollute the measurement.
use misaki_rs::{G2P, Language};
use serde::Deserialize;
use std::collections::HashMap;
#[derive(Deserialize)]
struct Case { text: String, ps: String, tokens: Vec<(String, Option<String>)>, tags: Vec<String> }
#[derive(Deserialize)]
struct Golden { cases: Vec<Case> }
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let g: Golden = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let mut e = G2P::new(Language::EnglishUS); e.unk = String::new();

    let (mut tok_ok, mut tags_tot, mut tags_ok) = (0usize, 0usize, 0usize);
    let (mut pass, mut fail_tagdiff, mut fail_logic) = (0usize, 0usize, 0usize);
    let mut confuse: HashMap<(String,String), usize> = HashMap::new();

    for c in &g.cases {
        let Ok((ps, toks)) = e.g2p(&c.text) else { continue };
        let same_shape = toks.len() == c.tokens.len()
            && toks.iter().zip(&c.tokens).all(|(a,b)| a.text == b.0);
        if !same_shape { continue; }       // tokenizer disagrees: excluded
        tok_ok += 1;
        let mut any_tag_diff = false;
        for (got, want) in toks.iter().zip(c.tags.iter()) {
            tags_tot += 1;
            if &got.tag == want { tags_ok += 1; } else {
                any_tag_diff = true;
                *confuse.entry((want.clone(), got.tag.clone())).or_insert(0) += 1;
            }
        }
        if ps == c.ps { pass += 1; }
        else if any_tag_diff { fail_tagdiff += 1; }
        else { fail_logic += 1; }
    }

    println!("cases where tokenization already agrees: {tok_ok}");
    println!("  tag accuracy vs spaCy : {tags_ok}/{tags_tot} ({:.1}%)", 100.0*tags_ok as f64/tags_tot.max(1) as f64);
    println!("\n  of those {tok_ok} cases:");
    println!("    phonemes already match        : {pass}");
    println!("    fail AND have a tag mismatch  : {fail_tagdiff}   <- tagger-attributable");
    println!("    fail with ALL TAGS CORRECT    : {fail_logic}   <- pure logic gap");
    let mut v: Vec<_> = confuse.into_iter().collect();
    v.sort_by(|a,b| b.1.cmp(&a.1));
    println!("\n  top tag confusions (spaCy -> ours):");
    for ((w,g),n) in v.iter().take(12) { println!("    {n:>5}  {w:<6} -> {g}"); }
}
