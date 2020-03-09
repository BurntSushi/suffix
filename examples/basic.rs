extern crate suffix;

use suffix::SuffixTable;

fn main() {
    let st = SuffixTable::new("the quick brown fox was quick.");
    assert_eq!(st.positions("quick"), &[4, 24]);
}
