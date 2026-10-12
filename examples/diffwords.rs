//! Per-WORD divergence census across the golden fixtures.
//! Prints which words misaki-rs gets wrong most often, so work is prioritised
//! by frequency in real text rather than by guesswork.
use misaki_rs::{G2P, Language};
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize)]
struct Case { text: String, ps: String, tokens: Vec<(String, Option<String>)> }
#[derive(Deserialize)]
struct Golden { cases: Vec<Case> }

fn main() {
    let path = std::env::args().nth(1).expect("usage: diffwords <golden.json>");
    let g: Golden = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let engine = G2P::new(Language::EnglishUS);

    let mut census: HashMap<(String, String, String), usize> = HashMap::new();
    let mut n_struct = 0usize;
    let mut bad_cases = 0usize;
    for c in &g.cases {
        let Ok((ps, toks)) = engine.g2p(&c.text) else { continue };
        if ps == c.ps { continue; }
        bad_cases += 1;
        if toks.len() != c.tokens.len() { n_struct += 1; continue; }
        for (got, want) in toks.iter().zip(c.tokens.iter()) {
            let gp = got.phonemes.clone().unwrap_or_default();
            let wp = want.1.clone().unwrap_or_default();
            if gp != wp {
                *census.entry((want.0.clone(), wp, gp)).or_insert(0) += 1;
            }
        }
    }
    let mut v: Vec<_> = census.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1));
    println!("{bad_cases} failing cases; {n_struct} differ in TOKEN COUNT (structural)");
    println!("\n{:>5}  {:<18} {:<24} {:<24}", "n", "word", "misaki 0.9.4", "misaki-rs");
    for ((w, want, got), n) in v.iter().take(30) {
        let t = |s: &String| -> String { s.chars().take(23).collect() };
        println!("{n:>5}  {:<18} {:<24} {:<24}", w.chars().take(17).collect::<String>(), t(want), t(got));
    }
}
