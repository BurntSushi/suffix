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
use std::iter;
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

pub struct SuffixArray {
    text: String,
    indices: Vec<u32>,
    lcp_lens: Vec<u32>,
}

impl SuffixArray {
    pub fn to_suffix_tree<'s>(&'s self) -> SuffixTree<'s> {
        to_suffix_tree::to_suffix_tree(self)
    }

    pub fn suffix(&self, i: uint) -> &str {
        sf!(self.text, self.indices[i])
    }

    pub fn lcp(&self, i: uint) -> &str {
        s!(self.text, self.indices[i], self.indices[i] + self.lcp_lens[i])
    }
}

impl fmt::Show for SuffixArray {
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
    root: Box<Node<'s>>,
}

struct Node<'s> {
    label: &'s str,
    children: BTreeMap<char, Box<Node<'s>>>,
    parent: Rawlink<Node<'s>>,
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
            root: box Node::new(s[0..0]),
        }
    }
}

impl<'s> Node<'s> {
    fn new(label: &'s str) -> Node<'s> {
        Node {
            label: label,
            children: BTreeMap::new(),
            parent: Rawlink::none(),
            terminal: true,
        }
    }

    fn parent(&self) -> Option<&Node<'s>> {
        self.parent.resolve_immut()
    }

    fn parent_mut(&mut self) -> Option<&mut Node<'s>> {
        self.parent.resolve()
    }

    fn children<'t>(&'t self) -> Children<'s, 't> {
        Children { it: self.children.values() }
    }

    fn ancestors<'t>(&'t self) -> Ancestors<'s, 't> {
        Ancestors { cur: Some(self) }
    }

    fn preorder<'t>(&'t self) -> Preorder<'s, 't> {
        Preorder::new(self)
    }

    fn leaves<'t>(&'t self) -> Leaves<'s, 't> {
        Leaves { it: self.preorder() }
    }

    fn is_terminal(&self) -> bool {
        self.terminal
    }

    fn is_root(&self) -> bool {
        self.parent.resolve_immut().is_none()
    }

    fn key(&self) -> char {
        self.label.char_at(0)
    }

    fn root_concat_len(&self) -> u32 {
        let mut len = self.label.len();
        let mut cur = self;
        loop {
            match cur.parent() {
                None => break,
                Some(p) => { cur = p; len += p.label.len(); }
            }
        }
        len as u32
    }

    fn depth(&self) -> uint {
        let mut depth = 0;
        let mut cur = self;
        loop {
            match cur.parent() {
                None => break,
                Some(p) => {
                    depth += 1;
                    cur = p;
                }
            }
        }
        depth
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
    fn resolve_immut<'a>(&self) -> Option<&'a T> {
        unsafe { self.p.as_ref() }
    }

    /// Convert the `Rawlink` into a mutable Option value.
    fn resolve<'a>(&mut self) -> Option<&'a mut T> {
        unsafe { self.p.as_mut() }
    }

    /// Return the `Rawlink` and replace with `Rawlink::none()`.
    fn take(&mut self) -> Rawlink<T> {
        mem::replace(self, Rawlink::none())
    }
}

impl<'s> fmt::Show for SuffixTree<'s> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(writeln!(f, "\n-----------------------------------------"));
        try!(writeln!(f, "SUFFIX TREE"));
        try!(writeln!(f, "text: {}", self.text));
        try!(self.root.fmt(f));
        writeln!(f, "-----------------------------------------")
    }
}

impl<'s> fmt::Show for Node<'s> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let indent: String = iter::repeat(' ').take(self.depth() * 2).collect();
        if self.is_root() {
            try!(writeln!(f, "ROOT"));
        } else {
            try!(writeln!(f, "{}{}", indent, self.label));
        }
        for ref node in self.children.values() {
            try!(node.fmt(f));
        }
        Ok(())
    }
}

struct Ancestors<'s: 't, 't> {
    cur: Option<&'t Node<'s>>,
}

impl<'s, 't> Iterator<&'t Node<'s>> for Ancestors<'s, 't> {
    fn next(&mut self) -> Option<&'t Node<'s>> {
        if let Some(node) = self.cur {
            self.cur = node.parent();
            Some(node)
        } else {
            None
        }
    }
}

struct Children<'s: 't, 't> {
    it: btree_map::Values<'t, char, Box<Node<'s>>>,
}

impl<'s, 't> Iterator<&'t Node<'s>> for Children<'s, 't> {
    fn next(&mut self) -> Option<&'t Node<'s>> {
        self.it.next().map(|n| &**n)
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        self.it.size_hint()
    }
}

impl<'s, 't> DoubleEndedIterator<&'t Node<'s>> for Children<'s, 't> {
    fn next_back(&mut self) -> Option<&'t Node<'s>> {
        self.it.next_back().map(|n| &**n)
    }
}

impl<'s, 't> ExactSizeIterator<&'t Node<'s>> for Children<'s, 't> {}

struct Preorder<'s: 't, 't> {
    stack: Vec<&'t Node<'s>>,
}

impl<'s, 't> Preorder<'s, 't> {
    fn new(start: &'t Node<'s>) -> Preorder<'s, 't> {
        Preorder { stack: vec![start] }
    }
}

impl<'s, 't> Iterator<&'t Node<'s>> for Preorder<'s, 't> {
    fn next(&mut self) -> Option<&'t Node<'s>> {
        match self.stack.pop() {
            None => None,
            Some(node) => {
                self.stack.extend(node.children().rev());
                Some(node)
            }
        }
    }
}

struct Leaves<'s: 't, 't> {
    it: Preorder<'s, 't>,
}

impl<'s, 't> Iterator<&'t Node<'s>> for Leaves<'s, 't> {
    fn next(&mut self) -> Option<&'t Node<'s>> {
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
