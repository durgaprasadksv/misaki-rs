use misaki_rs::{G2P, Language};
fn main() {
    let mut g = G2P::new(Language::EnglishUS); g.unk = String::new();
    for t in ["50%","1/2","23rd","1st","-40","$9"] {
        let (ps, toks) = g.g2p(t).unwrap();
        println!("  {t:<6} rs={ps:?}  tokens={:?}",
            toks.iter().map(|k| (k.text.as_str(), k.phonemes.clone())).collect::<Vec<_>>());
    }
}
