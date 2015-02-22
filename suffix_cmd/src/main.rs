#![allow(dead_code)]
#![feature(core, env, old_io, old_path)]

extern crate cbor;
extern crate docopt;
extern crate "rustc-serialize" as rustc_serialize;
extern crate suffix;

use std::borrow::ToOwned;
use std::env;
use std::error;
use std::fmt;
use std::old_io as io;

use docopt::Docopt;
use rustc_serialize::Decodable;

use db::DB;

macro_rules! cerr { ($tt:tt) => (return Err(Error::Other(format!($tt)))); }

macro_rules! lg {
    ($($arg:tt)*) => ({
        let _ = ::std::old_io::stderr().write_str(&*format!($($arg)*));
        let _ = ::std::old_io::stderr().write_str("\n");
    });
}

mod db;

static USAGE: &'static str = "
Usage:
    suffix index [ -f <index-file> ] <file> ...
    suffix search [ -f <index-file> ] <query>

Options:
    -f, --index <index-file>    A file path for the index to create/search.
";

#[derive(RustcDecodable)]
struct Args {
    cmd_index: bool,
    cmd_search: bool,
    arg_file: Vec<String>,
    arg_query: String,
    flag_index: Option<String>,
}

fn cmd_index(idx: Path, files: Vec<Path>) -> CliResult<()> {
    let mut enc = cbor::Encoder::from_writer(try!(io::File::create(&idx)));

    lg!("Reading documents into memory.");
    let mut documents = Vec::with_capacity(files.len());
    for file in files {
        documents.push((
            file.as_str().unwrap().to_owned(),
            try!(io::File::open(&file).read_to_string()),
        ));
    }

    lg!("Creating database.");
    let db = DB::create(documents);

    lg!("Writing database to disk.");
    Ok(try!(enc.encode(&[db])))
}

fn cmd_search(idx: Path, query: &str) -> CliResult<()> {
    lg!("Creating decoder");
    let mut dec = cbor::DirectDecoder::from_reader(try!(io::File::open(&idx)));
    lg!("Decoding index");
    // let db: DB = match dec.decode().next() {
        // None => cerr!("Empty index."),
        // Some(res) => try!(res),
    // };
    let db: DB = try!(Decodable::decode(&mut dec));
    lg!("Starting search");
    let mut results = db.search(query);
    results.sort();
    for r in results {
        println!("{}", r);
    }
    Ok(())
}

impl Args {
    fn run(&self) -> CliResult<()> {
        let idx = Path::new(&self.index());
        if self.cmd_index {
            cmd_index(idx, self.arg_file.iter().map(Path::new).collect())
        } else if self.cmd_search {
            cmd_search(idx, &self.arg_query)
        } else {
            // This case should be prevented by Docopt.
            unreachable!()
        }
    }

    fn index(&self) -> String {
        self.flag_index.clone().unwrap_or("suffix.db".to_string())
    }
}

type CliResult<T> = Result<T, Error>;

enum Error {
    Io(io::IoError),
    Cbor(cbor::CborError),
    Other(String),
}

impl error::FromError<io::IoError> for Error {
    fn from_error(err: io::IoError) -> Error { Error::Io(err) }
}

impl error::FromError<cbor::CborError> for Error {
    fn from_error(err: cbor::CborError) -> Error { Error::Cbor(err) }
}

impl error::FromError<String> for Error {
    fn from_error(err: String) -> Error { Error::Other(err) }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::Io(ref err) => err.fmt(f),
            Error::Cbor(ref err) => err.fmt(f),
            Error::Other(ref s) => write!(f, "{}", s),
        }
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
