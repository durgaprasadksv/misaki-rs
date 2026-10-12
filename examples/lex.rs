use misaki_rs::{Language, lexicon::Lexicon};
fn main() {
    let lx = Lexicon::new(Language::EnglishUS);
    for (w, tag) in [("Mr.","NNP"),("Mr","NNP"),("it's","PRP"),("I'm","PRP"),("'s","POS"),
                     ("five-storied","JJ"),("Caucus-Race","NNP"),("Austen","NNP")] {
        println!("  get_word({w:?}, {tag:?}) = {:?}", lx.get_word(w, tag, None, None));
        println!("     in_gold({w:?}) = {:?}", lx.in_gold(w));
    }
}
