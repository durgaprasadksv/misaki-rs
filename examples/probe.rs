use misaki_rs::{Language, lexicon::Lexicon, spacy_tagger::Tagger, spacy_tokenizer::SpacyTokenizer};
fn main() {
    let lx = Lexicon::new(Language::EnglishUS);
    println!("  rs lookup(ME,NNP)   = {:?}", lx.lookup("ME","NNP",None,None));
    println!("  rs is_known(ME,NNP) = {}", lx.is_known("ME","NNP"));
    let tk = SpacyTokenizer::new();
    let tg = Tagger::load();
    let text = "I\u{2019}m sure _I_ shan\u{2019}t be able!";
    let toks = tk.tokenize(text);
    let inp: Vec<(String,String,bool)> = toks.iter()
        .map(|(t,ws)| (t.clone(), misaki_rs::norms::norm_of(t, misaki_rs::spacy_tagger::hash_string), !ws.is_empty()))
        .collect();
    println!("  rs tags: {:?}", toks.iter().map(|t| t.0.as_str()).zip(tg.tag(&inp)).collect::<Vec<_>>());
}
