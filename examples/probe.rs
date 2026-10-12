use misaki_rs::{G2P, Language, lexicon::Lexicon, spacy_tagger::Tagger};
fn main() {
    let t = Tagger::load();
    let inp: Vec<(String,String,bool)> = ["I","have","three","PDFs","and","two","URLs","to","send","you","."]
        .iter().map(|w| (w.to_string(), w.to_lowercase(), true)).collect();
    println!("tags: {:?}", t.tag(&inp));
    let lx = Lexicon::new(Language::EnglishUS);
    for tag in ["NNS","NNPS","NNP","NN"] {
        println!("  get_word(PDFs,{tag}) = {:?}   stem_s = {:?}",
            lx.get_word("PDFs", tag, None, None), lx.stem_s("PDFs", tag, None, None));
    }
    let mut g = G2P::new(Language::EnglishUS); g.unk = String::new();
    let (ps, toks) = g.g2p("I have three PDFs and two URLs to send you.").unwrap();
    println!("\nrs [{ps}]");
    for k in &toks { println!("   {:<8?} tag={:<6?} ph={:?}", k.text, k.tag, k.phonemes); }
}
