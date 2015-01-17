//! Documentation for the `suffix` crate.

#![crate_name = "suffix"]
#![doc(html_root_url = "http://burntsushi.net/rustdoc/suffix")]
#![experimental]

#![allow(dead_code, unused_imports, unused_variables)]
#![allow(unstable)]

#[macro_use] extern crate log;
#[cfg(test)] extern crate quickcheck;

use std::collections::btree_map::{self, BTreeMap};
use std::collections::RingBuf;
use std::fmt;
use std::iter::{self, AdditiveIterator};
use std::mem;
use std::ptr;

pub use array::naive as array_naive;
pub use array::{naive_table, sais_table};
pub use array2::sais_table as sais_table2;

#[derive(Eq, PartialEq)]
pub struct SuffixArray<'s> {
    text: &'s str,
    table: Vec<u32>,
    inverse: Vec<u32>,
    lcp_lens: Vec<u32>,
}

impl<'s> SuffixArray<'s> {
    pub fn len(&self) -> u32 {
        self.text.len() as u32
    }

    pub fn to_suffix_tree(&'s self) -> SuffixTree<'s> {
        to_suffix_tree::to_suffix_tree(self)
    }

    pub fn suffix(&self, i: u32) -> &str {
        &self.text[self.table[i as usize] as usize..]
    }

    pub fn lcp(&self, i: u32) -> &str {
        let i = i as usize;
        let sufi = self.table[i] as usize;
        &self.text[sufi..sufi + (self.lcp_lens[i] as usize)]
    }
}

impl<'s> fmt::Show for SuffixArray<'s> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(writeln!(f, "\n-----------------------------------------"));
        try!(writeln!(f, "SUFFIX ARRAY"));
        try!(writeln!(f, "text: {}", self.text));
        for (rank, &sufstart) in self.table.iter().enumerate() {
            try!(writeln!(f, "suffix[{}] {}, {}",
                          rank, sufstart, self.suffix(rank as u32)));
        }
        for (sufstart, &rank) in self.inverse.iter().enumerate() {
            try!(writeln!(f, "inverse[{}] {}, {}",
                          sufstart, rank, self.suffix(rank as u32)));
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
    parent: Rawlink<Node>,
    children: BTreeMap<char, Box<Node>>,
    suffixes: Vec<u32>,
    start: u32,
    end: u32,
    path_len: u32,
}

struct Rawlink<T> {
    p: *mut T,
}

impl<T> Copy for Rawlink<T> {}

impl<'s> SuffixTree<'s> {
    fn init(s: &'s str) -> SuffixTree<'s> {
        SuffixTree {
            text: s,
            root: Node::leaf(0, 0, 0),
        }
    }

    fn root(&self) -> &Node {
        &*self.root
    }

    fn label(&self, node: &Node) -> &'s str {
        &self.text[node.start as usize .. node.end as usize]
    }

    fn suffix(&self, node: &Node) -> &'s str {
        assert!(node.suffixes.len() > 0);
        &self.text[node.suffixes[0] as usize..]
    }

    fn key(&self, node: &Node) -> char {
        self.label(node).char_at(0)
    }
}

impl Node {
    fn leaf(sufstart: u32, start: u32, end: u32) -> Box<Node> {
        Box::new(Node {
            parent: Rawlink::none(),
            children: BTreeMap::new(),
            suffixes: vec![sufstart],
            start: start,
            end: end,
            path_len: 0,
        })
    }

    fn internal(start: u32, end: u32) -> Box<Node> {
        Box::new(Node {
            parent: Rawlink::none(),
            children: BTreeMap::new(),
            suffixes: vec![],
            start: start,
            end: end,
            path_len: 0,
        })
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

    fn suffix_indices<'t>(&'t self) -> SuffixIndices<'t> {
        SuffixIndices { it: self.leaves(), node: None, cur_suffix: 0 }
    }

    fn has_terminals(&self) -> bool {
        self.suffixes.len() > 0
    }

    fn is_root(&self) -> bool {
        self.parent.resolve().is_none()
    }

    fn add_parent(&mut self, node: &mut Node) {
        self.parent = Rawlink::some(node);
        self.path_len = node.path_len + self.len();
    }

    fn depth(&self) -> usize {
        self.ancestors().count() - 1
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
                   node: &Node, depth: usize) -> fmt::Result {
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
                           terminals: {}, parent? {} }}",
               self.start, self.end, self.children.len(), self.suffixes.len(),
               self.parent().map(|_| "yes").unwrap_or("no"))
    }
}

struct Ancestors<'t> {
    cur: Option<&'t Node>,
}

impl<'t> Iterator for Ancestors<'t> {
    type Item = &'t Node;

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

impl<'t> Iterator for Children<'t> {
    type Item = &'t Node;

    fn next(&mut self) -> Option<&'t Node> {
        self.it.next().map(|n| &**n)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<'t> DoubleEndedIterator for Children<'t> {
    fn next_back(&mut self) -> Option<&'t Node> {
        self.it.next_back().map(|n| &**n)
    }
}

impl<'t> ExactSizeIterator for Children<'t> {}

struct Preorder<'t> {
    stack: Vec<&'t Node>,
}

impl<'t> Preorder<'t> {
    fn new(start: &'t Node) -> Preorder<'t> {
        Preorder { stack: vec![start] }
    }
}

impl<'t> Iterator for Preorder<'t> {
    type Item = &'t Node;

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

impl<'t> Iterator for Leaves<'t> {
    type Item = &'t Node;

    fn next(&mut self) -> Option<&'t Node> {
        for n in self.it {
            if n.has_terminals() {
                return Some(n);
            }
        }
        None
    }
}

struct SuffixIndices<'t> {
    it: Leaves<'t>,
    node: Option<&'t Node>,
    cur_suffix: u32,
}

impl<'t> Iterator for SuffixIndices<'t> {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        if let Some(node) = self.node {
            if (self.cur_suffix as usize) < node.suffixes.len() {
                self.cur_suffix += 1;
                return Some(node.suffixes[self.cur_suffix as usize - 1]);
            }
            self.node = None;
            self.cur_suffix = 0;
        }
        match self.it.next() {
            None => None,
            Some(leaf) => { self.node = Some(leaf); self.next() }
        }
    }
}

mod array;
mod array2;
mod to_suffix_tree;
