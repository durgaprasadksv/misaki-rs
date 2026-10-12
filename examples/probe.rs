use misaki_rs::{G2P, Language};
fn main() {
    let mut g = G2P::new(Language::EnglishUS); g.unk = String::new();
    for t in ["1-800-555-0199", "555-1234"] {
        let (ps, toks) = g.g2p(t).unwrap();
        println!("  {t:<16} [{ps}]");
        for k in &toks { println!("      {:<16?} tag={:<5?} ph={:?}", k.text, k.tag, k.phonemes); }
    }
}
