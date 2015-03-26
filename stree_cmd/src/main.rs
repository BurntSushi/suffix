#![feature(exit_status)]

extern crate docopt;
extern crate rustc_serialize;
extern crate suffix_tree;

use std::env;
use std::error;
use std::fmt;
use std::io::{self, Write};

use docopt::Docopt;
use suffix_tree::{SuffixTree, Node};

macro_rules! cerr { ($tt:tt) => (return Err(Error::Other(format!($tt)))); }

macro_rules! lg {
    ($($arg:tt)*) => ({ writeln!(&mut io::stderr(), $($arg)*).unwrap(); });
}

static USAGE: &'static str = "
Usage:
    stree [ <text> ... ]
    stree -h | --help

Options:
    -f, --index <index-file>    A file path for the index to create/search.
    -h, --help                  Show this usage message.
";

#[derive(RustcDecodable)]
struct Args {
    arg_text: Vec<String>,
}

type CliResult<T> = Result<T, Error>;

enum Error {
    Io(io::Error),
    Other(String),
}

impl error::FromError<io::Error> for Error {
    fn from_error(err: io::Error) -> Error { Error::Io(err) }
}

impl error::FromError<String> for Error {
    fn from_error(err: String) -> Error { Error::Other(err) }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::Io(ref err) => err.fmt(f),
            Error::Other(ref s) => write!(f, "{}", s),
        }
    }
}

fn main() {
    let args: Args = Docopt::new(USAGE)
                            .and_then(|d| d.decode())
                            .unwrap_or_else(|e| e.exit());
    if let Err(err) = args.run() {
        write!(&mut io::stderr(), "{}", err).unwrap();
        env::set_exit_status(1);
    }
}

impl Args {
    fn run(&self) -> CliResult<()> {
        print_dot_tree(&SuffixTree::new(self.text()));
        Ok(())
    }

    fn text(&self) -> String {
        self.arg_text.connect(" ")
    }
}

fn print_dot_tree(st: &SuffixTree) {
    println!("digraph tree {{");
    println!("label=<<FONT POINT-SIZE=\"20\">{}</FONT>>;", st.text());
    println!("labelloc=\"t\";");
    println!("labeljust=\"l\";");
    print_dot_node(st, st.root(), 0, 0);
    println!("}}");
}

fn print_dot_node(st: &SuffixTree, node: &Node, parent: u32, mut id: u32) -> u32 {
    // This entire function could be drastically simplified if we could nab
    // a unique id for each node.
    let node_id = id;
    id += 1;

    let label = if is_only_leaf(node) {
        println!("{} [label=\"{}\", shape=box]", node_id, terminals(node));
        format!("{}$", st.label(node))
    } else {
        println!("{} [label=\"\"]", node_id);
        if node.has_terminals() {
            println!("{} [label=\"{}\", shape=box]", id, terminals(node));
            println!("{} -> {} [label=\"$\"]", node_id, id);
            id += 1;
        }
        st.label(node).to_string()
    };
    if parent != node_id {
        println!("{} -> {} [label=\"{}\"];", parent, node_id, label);
    }
    for child in node.children() {
        id = print_dot_node(st, child, node_id, id);
    }
    id
}

fn is_only_leaf(node: &Node) -> bool {
    node.children().len() == 0 && node.suffixes().len() >= 1
}

fn terminals(node: &Node) -> String {
    node.suffixes().iter()
        .map(|&n| n.to_string()).collect::<Vec<_>>().connect(", ")
}
