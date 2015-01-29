#![allow(unstable)]

extern crate "rustc-serialize" as rustc_serialize;
extern crate docopt;
extern crate suffix;

use std::old_io::File;
use docopt::Docopt;
use suffix::SuffixTable;

static USAGE: &'static str = "
Usage:
    suffix naive <file>
    suffix sais <file>
";

#[derive(RustcDecodable)]
struct Args {
    arg_file: String,
    cmd_naive: bool,
    cmd_sais: bool,
}

fn main() {
    let args: Args = Docopt::new(USAGE).and_then(|d| d.decode())
                                       .unwrap_or_else(|e| e.exit());
    println!("reading file...");
    let data = File::open(&Path::new(args.arg_file)).read_to_string().unwrap();
    println!("... done reading file.");
    let data = data.trim();
    println!("data length: {}", data.len());
    if args.cmd_naive {
        SuffixTable::new_naive(data);
    } else if args.cmd_sais {
        SuffixTable::new(data);
    } else {
        unreachable!();
    }
}
