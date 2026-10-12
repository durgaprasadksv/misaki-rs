use misaki_rs::{G2P, Language, lexicon::Lexicon};
fn main() {
    let lx = Lexicon::new(Language::EnglishUS);
    println!("  get_word(minus,NN) = {:?}", lx.get_word("minus","NN",None,None));
    let mut g = G2P::new(Language::EnglishUS); g.unk = String::new();
    for t in ["-40", "The temperature was -40 degrees.", "555-1234"] {
        let (ps, toks) = g.g2p(t).unwrap();
        println!("  {t:<34} [{ps}]");
        for k in &toks { println!("      {:<12?} tag={:<6?} ph={:?}", k.text, k.tag, k.phonemes); }
    }
}
