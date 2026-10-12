use misaki_rs::{G2P, Language};
fn main() {
    let mut g = G2P::new(Language::EnglishUS); g.unk = String::new();
    for t in ["[Illustration: GEORGE ALLEN]", "an out-of-the-way place", "the passers-by saw"] {
        let (ps, toks) = g.g2p(t).unwrap();
        println!("{t}\n  rs [{ps}]");
        for k in &toks { println!("      {:<18?} tag={:<7?} ph={:?}", k.text, k.tag, k.phonemes); }
    }
}
