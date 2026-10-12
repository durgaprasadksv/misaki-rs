//! Cases where the phoneme STRING matches but per-token attribution does not.
//! These are the dangerous ones: identical audio, different word timestamps.
use misaki_rs::{G2P, Language};
use serde::Deserialize;
#[derive(Deserialize)]
struct Case { text: String, ps: String, tokens: Vec<(String, Option<String>)> }
#[derive(Deserialize)]
struct Golden { cases: Vec<Case> }
fn main() {
    let g: Golden = serde_json::from_str(&std::fs::read_to_string(std::env::args().nth(1).unwrap()).unwrap()).unwrap();
    let mut e = G2P::new(Language::EnglishUS); e.unk = String::new();
    let mut n = 0;
    for c in &g.cases {
        let Ok((ps, toks)) = e.g2p(&c.text) else { continue };
        if ps != c.ps { continue; }
        let got: Vec<(String, Option<String>)> = toks.iter().map(|t| (t.text.clone(), t.phonemes.clone())).collect();
        if got == c.tokens { continue; }
        n += 1;
        if n > 5 { continue; }
        println!("{}", c.text.chars().take(66).collect::<String>());
        let mut i = 0;
        while i < got.len() && i < c.tokens.len() && got[i] == c.tokens[i] { i += 1; }
        println!("   first divergence at token {i} ({} vs {} tokens)", c.tokens.len(), got.len());
        for k in i..(i+3).min(c.tokens.len().max(got.len())) {
            println!("     misaki {:?}", c.tokens.get(k));
            println!("     rs     {:?}", got.get(k));
        }
    }
    println!("\n{n} cases: string matches, token attribution differs");
}
