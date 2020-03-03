extern crate suffix;

use suffix::SuffixTable;

fn main() {
    // build a suffix table st from a string
    let st = SuffixTable::new("the quick brown fox was quick.");
    // This is what a suffix table looks like!
    print!("{:?}", st);

    // If we want to find the substring "quick" then we should get
    // two results back. The first is the 4th index, the 2nd is at 
    // the 24th index of the original string.
    let result =  st.positions("quick");
    println!("search result: {:?}", result);
    assert_eq!(result, &[4, 24]);

    // print the contents of the result
    for i in result {
        println!("quick found! Starts at index: {}", i);
    }
}