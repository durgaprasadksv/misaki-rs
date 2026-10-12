use misaki_rs::{G2P, Language};
fn main() {
    let g = G2P::new(Language::EnglishUS);
    for t in ["It's fine", "hadn't gone", "dogs' bones", "Thomas\u{2019} books",
              "_like that_", "_it connects", "Mr. Smith", "five-storied house"] {
        let toks = g.tokenize(t);
        println!("  {:<22} {:?}", t,
            toks.iter().map(|k| (k.text.as_str(), k.whitespace.as_str())).collect::<Vec<_>>());
    }
}
