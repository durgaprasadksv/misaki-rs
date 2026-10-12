use misaki_rs::{G2P, Language};
fn main() {
    let mut g = G2P::new(Language::EnglishUS); g.unk = String::new();
    for t in ["\u{2018}How doth the little\u{2019}", "_I_ am here", "an out-of-the-way place", "the passers-by saw"] {
        let (ps, toks) = g.g2p(t).unwrap();
        println!("{t}\n  rs [{ps}]");
        for k in &toks { println!("      {:<18?} tag={:<6?} ph={:?}", k.text, k.tag, k.phonemes); }
    }
}
