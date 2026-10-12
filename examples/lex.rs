use misaki_rs::{Language, lexicon::Lexicon};
fn main() {
    let lx = Lexicon::new(Language::EnglishUS);
    println!("  is_known(\"dog\",NNS)   = {}", lx.is_known("dog","NNS"));
    println!("  is_known(\"dog's\",NNS) = {}", lx.is_known("dog's","NNS"));
    println!("  is_known(\"dogs\",NNS)  = {}", lx.is_known("dogs","NNS"));
    println!("  in_gold(\"dog's\")      = {}", lx.in_gold("dog's"));
    println!("  lookup(\"dog's\",NNS)   = {:?}", lx.lookup("dog's","NNS",None,None));
    println!("  lookup(\"dogs\",NNS)    = {:?}", lx.lookup("dogs","NNS",None,None));
    println!("  stem_s(\"dogs'\",NNS)   = {:?}", lx.stem_s("dogs'","NNS",None,None));
}
