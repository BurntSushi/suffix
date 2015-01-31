//! Documentation for the `suffix` crate.

#![crate_name = "suffix"]
#![doc(html_root_url = "http://burntsushi.net/rustdoc/suffix")]

// #![allow(dead_code, unused_imports, unused_variables)]
#![allow(unused_features)] // for `test`
#![feature(collections, core, test)]

#[macro_use] extern crate log;
#[cfg(test)] extern crate quickcheck;
#[cfg(test)] extern crate test;

pub use array::SuffixArray;
pub use table::SuffixTable;
pub use tree::{
    Node, SuffixTree,
    Children, Ancestors, Preorder, Leaves, SuffixTreeIndices,
};

fn vec_from_elem<T: Copy>(len: usize, init: T) -> Vec<T> {
    let mut vec: Vec<T> = Vec::with_capacity(len);
    unsafe { vec.set_len(len); }
    for v in vec.iter_mut() { *v = init; }
    vec
}

mod array;
mod table;
mod tree;
