use misaki_rs::{G2P, Language};
fn main() {
    let g = G2P::new(Language::EnglishUS);
    for t in ["man's heart", "man\u{2019}s heart", "hadn\u{2019}t cried", "o\u{2019}clock now", "dogs' bones"] {
        println!("  {:<18} {:?}", t, g.tokenize(t).iter().map(|k| k.text.clone()).collect::<Vec<_>>());
    }
}
