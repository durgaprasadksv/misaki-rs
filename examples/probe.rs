use misaki_rs::{G2P, Language};
fn main() {
    let mut g = G2P::new(Language::EnglishUS); g.unk = String::new();
    for t in ["Mr. Smith came.", "it\u{2019}s fine", "I\u{2019}m here", "a woman\u{2019}s hat", "Austen\u{2019}s book"] {
        let (ps, toks) = g.g2p(t).unwrap();
        println!("{t}\n  ps [{ps}]");
        for k in &toks { println!("     {:<12?} tag={:<5?} ph={:?}", k.text, k.tag, k.phonemes); }
    }
}
