#![allow(unstable)]

extern crate "rustc-serialize" as rustc_serialize;
extern crate docopt;
extern crate suffix;

use std::io::File;
use docopt::Docopt;
use suffix::{naive_table, sais_table, sais_table2, sais2};

static USAGE: &'static str = "
Usage:
    suffix naive <file>
    suffix sais <file>
    suffix sais2 <file>
";

#[derive(RustcDecodable)]
struct Args {
    arg_file: String,
    cmd_naive: bool,
    cmd_sais: bool,
    cmd_sais2: bool,
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
        naive_table(data);
    } else if args.cmd_sais {
        sais_table(data);
    } else if args.cmd_sais2 {
        sais2(data);
    } else {
        unreachable!();
    }
}
