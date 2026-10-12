use misaki_rs::{G2P, Language};
fn main() {
    let mut g = G2P::new(Language::EnglishUS); g.unk = String::new();
    for t in ["a Caucus-Race here","prize-money paid","Austen\u{2019}s book","It\u{2019}s fine","Good-bye now","the eighteenth-century poet"] {
        let (ps, toks) = g.g2p(t).unwrap();
        println!("{t}\n  rs [{ps}]");
        for k in &toks { println!("      {:<22?} ph={:?}", k.text, k.phonemes); }
    }
}
