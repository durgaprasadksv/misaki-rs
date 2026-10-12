//! Census of TOKEN COUNT mismatches -- the structural failures.
use misaki_rs::{G2P, Language};
use serde::Deserialize;
use std::collections::HashMap;
#[derive(Deserialize)]
struct Case { text: String, ps: String, tokens: Vec<(String, Option<String>)> }
#[derive(Deserialize)]
struct Golden { cases: Vec<Case> }
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let g: Golden = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let mut e = G2P::new(Language::EnglishUS); e.unk = String::new();
    let mut shapes: HashMap<String, usize> = HashMap::new();
    let mut shown = 0;
    for c in &g.cases {
        let Ok((_, toks)) = e.g2p(&c.text) else { continue };
        if toks.len() == c.tokens.len() { continue; }
        // find the first index where the token TEXT diverges
        let mut i = 0;
        while i < toks.len() && i < c.tokens.len() && toks[i].text == c.tokens[i].0 { i += 1; }
        let want: Vec<&str> = c.tokens.iter().skip(i).take(3).map(|t| t.0.as_str()).collect();
        let got: Vec<&str> = toks.iter().skip(i).take(3).map(|t| t.text.as_str()).collect();
        *shapes.entry(format!("misaki {:?}  vs  rs {:?}", want, got)).or_insert(0) += 1;
        if shown < 6 {
            println!("{}", c.text.chars().take(70).collect::<String>());
            println!("   misaki {} tokens, rs {} tokens; first divergence at {i}", c.tokens.len(), toks.len());
            println!("     misaki {:?}", c.tokens.iter().skip(i.saturating_sub(1)).take(5).map(|t| t.0.as_str()).collect::<Vec<_>>());
            println!("     rs     {:?}", toks.iter().skip(i.saturating_sub(1)).take(5).map(|t| t.text.as_str()).collect::<Vec<_>>());
            shown += 1;
        }
    }
    let mut v: Vec<_> = shapes.into_iter().collect();
    v.sort_by(|a,b| b.1.cmp(&a.1));
    println!("\n{:>5}  divergence shape", "n");
    for (k,n) in v.iter().take(12) { println!("{n:>5}  {k}"); }
}
