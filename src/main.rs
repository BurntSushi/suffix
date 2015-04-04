use std::env;
use std::fs;
use std::io::Read;

#[allow(dead_code)]
mod table;
#[allow(dead_code)]
mod unicode;

fn main() {
    let fpath = env::args().nth(1).unwrap();
    let mut text = String::new();
    fs::File::open(fpath).unwrap().read_to_string(&mut text).unwrap();

    let st = table::SuffixTable::new(text);
    println!("Suffixes: {}", st.len());
}
