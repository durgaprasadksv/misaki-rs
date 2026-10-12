use misaki_rs::{G2P, Language};
fn main() {
    let mut g = G2P::new(Language::EnglishUS); g.unk = String::new();
    for t in ["3.5", "26.2", "-40", "G2P", "\"don't go\"", "'Single quotes'"] {
        let (ps, toks) = g.g2p(t).unwrap();
        println!("  {t:<16} [{ps}]");
        for k in &toks { println!("        {:<10?} tag={:<6?} ph={:?}", k.text, k.tag, k.phonemes); }
    }
}
