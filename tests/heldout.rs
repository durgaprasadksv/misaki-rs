//! A HELD-OUT check against misaki 0.9.4.
//!
//! The 771 cases in `tests/parity.rs` were also what this port was developed
//! against, so they cannot measure overfitting. This corpus -- 900 sentences
//! from five books never used while fixing anything -- can, and on first run it
//! found 8 divergences the committed fixtures did not contain.
//!
//! It asserts a REGRESSION FLOOR rather than exactness, because a known gap
//! remains. The floor exists so the number can only move one way; it is not a
//! target, and the comment below says what has to change to retire it.
//!
//! ## The remaining gap
//!
//! misaki derives its `TokenContext` from the phonemes of the merged WORD,
//! because its resolution loop iterates over adjacency groups. Ours iterates
//! over subtokens, so a word whose merged phonemes differ from its parts' gives
//! the wrong lookahead:
//!
//! ```text
//!     "I belong to Tellson’s Bank."
//!       misaki  to -> tə      (merged `Tellson’s` is `s`, a consonant)
//!       ours    to -> tʊ      (the `s` SUBTOKEN is the letter name ˈɛs, a vowel)
//! ```
//!
//! Closing it means resolving group by group in the reverse pass and updating
//! the rolling context from each group's merged phonemes -- misaki's own
//! structure, and a restructure of `g2p` rather than another patch.
use misaki_rs::{G2P, Language};
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    text: String,
    ps: String,
    tokens: Vec<(String, Option<String>)>,
}
#[derive(Deserialize)]
struct Golden {
    cases: Vec<Case>,
}

/// Raise this as the gap closes; never lower it.
const FLOOR: usize = 895;

#[test]
fn heldout_regression_floor() {
    let raw = std::fs::read_to_string("tests/fixtures/golden_heldout_us.json").unwrap();
    let g: Golden = serde_json::from_str(&raw).unwrap();
    let mut e = G2P::new(Language::EnglishUS);
    e.unk = String::new();

    let mut ok = 0usize;
    let mut shown = 0;
    for c in &g.cases {
        let Ok((ps, toks)) = e.g2p(&c.text) else { continue };
        let got: Vec<(String, Option<String>)> =
            toks.iter().map(|t| (t.text.clone(), t.phonemes.clone())).collect();
        if ps == c.ps && got == c.tokens {
            ok += 1;
        } else if shown < 8 {
            shown += 1;
            println!("  {}", c.text.chars().take(72).collect::<String>());
        }
    }
    println!(
        "\nheld-out: {}/{} exact ({:.1}%), floor {}",
        ok,
        g.cases.len(),
        100.0 * ok as f64 / g.cases.len() as f64,
        FLOOR
    );
    assert!(ok >= FLOOR, "held-out parity regressed below {FLOOR}: {ok}");
}
