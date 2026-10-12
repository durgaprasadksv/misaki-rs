use misaki_rs::{G2P, Language};
fn main() {
    let mut g = G2P::new(Language::EnglishUS); g.unk = String::new();
    for t in ["a G2P engine", "G2P"] {
        let (ps, toks) = g.g2p(t).unwrap();
        println!("  {t:<14} [{ps}]");
        for k in &toks { println!("      {:<8?} tag={:<6?} alias={:?} ph={:?}",
            k.text, k.tag, k.underscore().alias, k.phonemes); }
    }
}
