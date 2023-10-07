#![allow(deprecated)]

extern crate docopt;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate suffix_tree;

use std::fmt;
use std::io::{self, Write};

use docopt::Docopt;
use suffix_tree::{Node, SuffixTree};

static USAGE: &str = "
Usage:
    stree [ <text> ... ]
    stree -h | --help

Options:
    -f, --index <index-file>    A file path for the index to create/search.
    -h, --help                  Show this usage message.
";

#[derive(Deserialize)]
struct Args {
    arg_text: Vec<String>,
}

type CliResult<T> = Result<T, Error>;

enum Error {
    Io(io::Error),
    Other(String),
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Error {
        Error::Io(err)
    }
}

impl From<String> for Error {
    fn from(err: String) -> Error {
        Error::Other(err)
    }
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
        .and_then(|d| d.deserialize())
        .unwrap_or_else(|e| e.exit());
    if let Err(err) = args.run() {
        write!(&mut io::stderr(), "{}", err).unwrap();
        ::std::process::exit(1);
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

fn print_dot_node(
    st: &SuffixTree,
    node: &Node,
    parent: u32,
    mut id: u32,
) -> u32 {
    // This entire function could be drastically simplified if we could nab
    // a unique id for each node.
    let node_id = id;
    id += 1;

    let label = if is_only_leaf(node) {
        println!("{} [label=\"{}\", shape=box]", node_id, terminals(node));
        format!("{}$", label(st, node))
    } else {
        println!("{} [label=\"\"]", node_id);
        if node.has_terminals() {
            println!("{} [label=\"{}\", shape=box]", id, terminals(node));
            println!("{} -> {} [label=\"$\"]", node_id, id);
            id += 1;
        }
        label(st, node)
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
    node.children().len() == 0 && !node.suffixes().is_empty()
}

fn terminals(node: &Node) -> String {
    node.suffixes()
        .iter()
        .map(|&n| n.to_string())
        .collect::<Vec<_>>()
        .connect(", ")
}

fn label(st: &SuffixTree, node: &Node) -> String {
    let bytes = st.label(node);
    match String::from_utf8(bytes.to_vec()) {
        Ok(s) => s,
        Err(_) => format!("{:?}", bytes),
    }
}
