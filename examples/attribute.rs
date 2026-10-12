//! Precise attribution: for each token whose PHONEMES differ, did its TAG also
//! differ? A tag mismatch on a token whose phonemes are already correct is
//! harmless (our punctuation path is character-driven, not tag-driven), so
//! counting those overstates the tagger's role.
use misaki_rs::{G2P, Language};
use serde::Deserialize;
use std::collections::HashMap;
#[derive(Deserialize)]
struct Case { text: String, tokens: Vec<(String, Option<String>)>, tags: Vec<String> }
#[derive(Deserialize)]
struct Golden { cases: Vec<Case> }
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let g: Golden = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let mut e = G2P::new(Language::EnglishUS); e.unk = String::new();
    let (mut ph_bad, mut ph_bad_tag_bad, mut ph_bad_tag_ok) = (0usize, 0usize, 0usize);
    let mut culprits: HashMap<(String,String,String), usize> = HashMap::new();
    for c in &g.cases {
        let Ok((_, toks)) = e.g2p(&c.text) else { continue };
        if toks.len() != c.tokens.len() { continue; }
        if !toks.iter().zip(&c.tokens).all(|(a,b)| a.text == b.0) { continue; }
        for ((got, want), wtag) in toks.iter().zip(&c.tokens).zip(&c.tags) {
            let gp = got.phonemes.clone().unwrap_or_default();
            let wp = want.1.clone().unwrap_or_default();
            if gp == wp { continue; }
            ph_bad += 1;
            if &got.tag != wtag {
                ph_bad_tag_bad += 1;
                *culprits.entry((want.0.clone(), wtag.clone(), got.tag.clone())).or_insert(0) += 1;
            } else {
                ph_bad_tag_ok += 1;
                *culprits.entry((want.0.clone(), format!("={}", wtag), "TAG-OK".into())).or_insert(0) += 1;
            }
        }
    }
    println!("tokens with WRONG PHONEMES: {ph_bad}");
    println!("  of which tag ALSO wrong : {ph_bad_tag_bad}  <- tagger could explain these");
    println!("  of which tag was correct: {ph_bad_tag_ok}  <- NOT the tagger");
    let mut v: Vec<_> = culprits.into_iter().collect();
    v.sort_by(|a,b| b.1.cmp(&a.1));
    println!("\n{:>5}  {:<16} {:<10} {:<10}", "n", "word", "spaCy tag", "our tag");
    for ((w,wt,gt),n) in v.iter().take(18) {
        println!("{n:>5}  {:<16} {wt:<10} {gt:<10}", w.chars().take(15).collect::<String>());
    }
}
