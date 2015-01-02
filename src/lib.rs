//! Documentation for the `suffix` crate.

#![crate_name = "suffix"]
#![doc(html_root_url = "http://burntsushi.net/rustdoc/suffix")]
#![feature(macro_rules, phase, slicing_syntax)]
#![experimental]

#![allow(dead_code, unused_imports, unused_variables)]

#[phase(plugin, link)] extern crate log;
#[cfg(test)] extern crate quickcheck;

use std::collections::btree_map::{mod, BTreeMap};
use std::collections::RingBuf;
use std::fmt;
use std::iter::{mod, AdditiveIterator};
use std::mem;
use std::ptr;

pub use array_naive::construct as array_naive;

// slice
macro_rules! s {
    ($e:expr, $from:expr, $to:expr) => ($e[$from as uint .. $to as uint]);
}

// slice to
macro_rules! st {
    ($e:expr, $to:expr) => ($e[.. $to as uint]);
}

// slice from
macro_rules! sf {
    ($e:expr, $from:expr) => ($e[$from as uint ..]);
}

pub struct SuffixArray<'s> {
    text: &'s str,
    indices: Vec<u32>,
    lcp_lens: Vec<u32>,
}

impl<'s> SuffixArray<'s> {
    pub fn to_suffix_tree(&'s self) -> SuffixTree<'s> {
        to_suffix_tree::to_suffix_tree(self)
    }

    pub fn suffix(&self, i: uint) -> &str {
        sf!(self.text, self.indices[i])
    }

    pub fn lcp(&self, i: uint) -> &str {
        s!(self.text, self.indices[i], self.indices[i] + self.lcp_lens[i])
    }
}

impl<'s> fmt::Show for SuffixArray<'s> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(writeln!(f, "\n-----------------------------------------"));
        try!(writeln!(f, "SUFFIX ARRAY"));
        try!(writeln!(f, "text: {}", self.text));
        for (i, &sufstart) in self.indices.iter().enumerate() {
            try!(writeln!(f, "suffix[{}:] {}", sufstart, self.suffix(i)));
        }
        for (i, &len) in self.lcp_lens.iter().enumerate() {
            try!(writeln!(f, "lcp_length[{}] {}", i, len));
        }
        writeln!(f, "-----------------------------------------")
    }
}

pub struct SuffixTree<'s> {
    text: &'s str,
    root: Box<Node>,
}

struct Node {
    start: u32,
    end: u32,
    path_len: u32,
    children: BTreeMap<char, Box<Node>>,
    parent: Rawlink<Node>,
    terminal: bool,
}

struct Rawlink<T> {
    p: *mut T,
}

impl<T> Copy for Rawlink<T> {}

impl<'s> SuffixTree<'s> {
    fn init(s: &'s str) -> SuffixTree<'s> {
        SuffixTree {
            text: s,
            root: Node::leaf(0, 0),
        }
    }

    fn root(&self) -> &Node {
        &*self.root
    }

    fn label(&self, node: &Node) -> &'s str {
        self.text[node.start as uint .. node.end as uint]
    }

    fn key(&self, node: &Node) -> char {
        self.label(node).char_at(0)
    }
}

impl Node {
    fn leaf(start: u32, end: u32) -> Box<Node> {
        box Node {
            start: start,
            end: end,
            path_len: 0,
            children: BTreeMap::new(),
            parent: Rawlink::none(),
            terminal: true,
        }
    }

    fn internal(start: u32, end: u32) -> Box<Node> {
        let mut node = Node::leaf(start, end);
        node.terminal = false;
        node
    }

    fn len(&self) -> u32 {
        self.end - self.start
    }

    fn parent(&self) -> Option<&Node> {
        self.parent.resolve()
    }

    fn parent_mut(&mut self) -> Option<&mut Node> {
        self.parent.resolve_mut()
    }

    fn children<'t>(&'t self) -> Children<'t> {
        Children { it: self.children.values() }
    }

    fn ancestors<'t>(&'t self) -> Ancestors<'t> {
        Ancestors { cur: Some(self) }
    }

    fn preorder<'t>(&'t self) -> Preorder<'t> {
        Preorder::new(self)
    }

    fn leaves<'t>(&'t self) -> Leaves<'t> {
        Leaves { it: self.preorder() }
    }

    fn is_terminal(&self) -> bool {
        self.terminal
    }

    fn is_root(&self) -> bool {
        self.parent.resolve().is_none()
    }

    fn add_parent(&mut self, node: &mut Node) {
        self.parent = Rawlink::some(node);
        self.path_len = node.path_len + self.len();
    }

    fn depth(&self) -> u32 {
        (self.ancestors().count() - 1) as u32
    }
}

/// Rawlink is a type like Option<T> but for holding a raw mutable pointer.
impl<T> Rawlink<T> {
    /// Like `Option::None` for Rawlink.
    fn none() -> Rawlink<T> {
        Rawlink{p: ptr::null_mut()}
    }

    /// Like `Option::Some` for Rawlink
    fn some(n: &mut T) -> Rawlink<T> {
        Rawlink{p: n}
    }

    /// Convert the `Rawlink` into an immutable Option value.
    fn resolve<'a>(&self) -> Option<&'a T> {
        unsafe { self.p.as_ref() }
    }

    /// Convert the `Rawlink` into a mutable Option value.
    fn resolve_mut<'a>(&mut self) -> Option<&'a mut T> {
        unsafe { self.p.as_mut() }
    }

    /// Return the `Rawlink` and replace with `Rawlink::none()`.
    fn take(&mut self) -> Rawlink<T> {
        mem::replace(self, Rawlink::none())
    }
}

impl<'s> fmt::Show for SuffixTree<'s> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fn fmt<'s>(f: &mut fmt::Formatter, st: &SuffixTree<'s>,
                   node: &Node, depth: uint) -> fmt::Result {
            let indent: String = iter::repeat(' ').take(depth * 2).collect();
            if node.is_root() {
                try!(writeln!(f, "ROOT"));
            } else {
                try!(writeln!(f, "{}{}", indent, st.label(node)));
            }
            for child in node.children() {
                try!(fmt(f, st, child, depth + 1));
            }
            Ok(())
        }
        try!(writeln!(f, "\n-----------------------------------------"));
        try!(writeln!(f, "SUFFIX TREE"));
        try!(writeln!(f, "text: {}", self.text));
        try!(fmt(f, self, self.root(), 0));
        writeln!(f, "-----------------------------------------")
    }
}

impl fmt::Show for Node {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Node {{ start: {}, end: {}, len(children): {}, \
                           terminal? {}, parent? {} }}",
               self.start, self.end, self.children.len(), self.terminal,
               self.parent.resolve().map(|_| "yes").unwrap_or("no"))
    }
}

struct Ancestors<'t> {
    cur: Option<&'t Node>,
}

impl<'t> Iterator<&'t Node> for Ancestors<'t> {
    fn next(&mut self) -> Option<&'t Node> {
        if let Some(node) = self.cur {
            self.cur = node.parent();
            Some(node)
        } else {
            None
        }
    }
}

struct Children<'t> {
    it: btree_map::Values<'t, char, Box<Node>>,
}

impl<'t> Iterator<&'t Node> for Children<'t> {
    fn next(&mut self) -> Option<&'t Node> {
        self.it.next().map(|n| &**n)
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        self.it.size_hint()
    }
}

impl<'t> DoubleEndedIterator<&'t Node> for Children<'t> {
    fn next_back(&mut self) -> Option<&'t Node> {
        self.it.next_back().map(|n| &**n)
    }
}

impl<'t> ExactSizeIterator<&'t Node> for Children<'t> {}

struct Preorder<'t> {
    stack: Vec<&'t Node>,
}

impl<'t> Preorder<'t> {
    fn new(start: &'t Node) -> Preorder<'t> {
        Preorder { stack: vec![start] }
    }
}

impl<'t> Iterator<&'t Node> for Preorder<'t> {
    fn next(&mut self) -> Option<&'t Node> {
        match self.stack.pop() {
            None => None,
            Some(node) => {
                self.stack.extend(node.children().rev());
                Some(node)
            }
        }
    }
}

struct Leaves<'t> {
    it: Preorder<'t>,
}

impl<'t> Iterator<&'t Node> for Leaves<'t> {
    fn next(&mut self) -> Option<&'t Node> {
        for n in self.it {
            if n.is_terminal() {
                return Some(n);
            }
        }
        None
    }
}

mod array_naive;
mod to_suffix_tree;
