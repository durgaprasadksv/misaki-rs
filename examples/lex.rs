use misaki_rs::{Language, lexicon::Lexicon};
fn main() {
    let lx = Lexicon::new(Language::EnglishUS);
    for w in ["PDF","PDFs","CEOs","CDs"] {
        println!("  {w:<6} is_known={:<5} in_gold={:<5}", lx.is_known(w,"NNP"), lx.in_gold(w));
        println!("         lookup   = {:?}", lx.lookup(w,"NNP",None,None));
        println!("         stem_s   = {:?}", lx.stem_s(w,"NNS",None,None));
        println!("         get_word = {:?}", lx.get_word(w,"NNS",None,None));
    }
}
