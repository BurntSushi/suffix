#![allow(dead_code)]
#![feature(env, io)]

extern crate docopt;
extern crate "rustc-serialize" as rustc_serialize;
extern crate suffix;

use std::env;
use std::old_io as io;

use docopt::Docopt;

static USAGE: &'static str = "
Usage:
    suffix index <index-out> <file> ...
    suffix search <query> <index> ...
";

#[derive(RustcDecodable)]
struct Args {
    cmd_index: bool,
    cmd_search: bool,
    arg_file: Vec<String>,
    arg_query: String,
    arg_index_out: String,
    arg_index: Vec<String>,
}

impl Args {
    fn run(&self) -> Result<(), String> {
        Ok(())
    }
}

fn main() {
    let args: Args = Docopt::new(USAGE)
                            .and_then(|d| d.decode())
                            .unwrap_or_else(|e| e.exit());
    if let Err(err) = args.run() {
        io::stderr().write_str(&format!("{}", err)).unwrap();
        env::set_exit_status(1);
    }
}
