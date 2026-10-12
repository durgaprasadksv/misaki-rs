use misaki_rs::{G2P, Language};
fn main() {
    let g = G2P::new(Language::EnglishUS);
    for t in ["He said \u{201c}go\u{201d} now.", "It\u{2019}s fine.", "A (paren) here.", "Bennet arrived."] {
        let (_, toks) = g.g2p(t).unwrap();
        println!("{t}");
        for k in &toks {
            println!("    text={:<10?} tag={:<8?} ph={:?}", k.text, k.tag, k.phonemes);
        }
    }
}
