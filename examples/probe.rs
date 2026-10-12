use misaki_rs::{G2P, Language, spacy_tagger::Tagger};
fn main() {
    let t = Tagger::load();
    let inp: Vec<(String,String,bool)> = vec![
        ("ladies".into(),"ladies".into(),false),
        ("\u{2019}".into(),"\u{2019}".into(),true),
        ("battle".into(),"battle".into(),true),
        ("with".into(),"with".into(),false)];
    println!("tagger on spaCy-like tokens: {:?}", t.tag(&inp));
    let mut g = G2P::new(Language::EnglishUS); g.unk = String::new();
    let (ps, toks) = g.g2p("ladies\u{2019} battle with").unwrap();
    println!("rs [{ps}]");
    for k in &toks { println!("   {:<10?} tag={:<6?} ph={:?}", k.text, k.tag, k.phonemes); }
}
