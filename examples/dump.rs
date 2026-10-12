use misaki_rs::{G2P, Language};
fn main() {
    let g = G2P::new(Language::EnglishUS);
    for t in ["A dog barked.", "Wait... what? No!", "The cat's toys; the dogs' bones."] {
        let (ps, toks) = g.g2p(t).unwrap();
        println!("text  {t}");
        println!("  ps  [{ps}]");
        for k in &toks {
            println!("      text={:<12?} ws={:<4?} ph={:?}", k.text, k.whitespace, k.phonemes);
        }
    }
}
