//! This crate provides fast suffix array construction with Unicode support.
//! There is some experimental support for suffix trees (only by constructing
//! the suffix array first).
//!
//! The details of the construction algorithm are documented on the
//! `SuffixArray` type.
//!
//! In general, suffix arrays are useful when you want to query some text
//! repeatedly and quickly.
//!
//! # Examples
//!
//! Usage is extremely simple. You just create the suffix array and then
//! search:
//!
//! ```rust
//! use suffix::SuffixTable;
//!
//! let st = SuffixTable::new("the quick brown fox was quick.");
//! assert_eq!(st.positions("quick"), vec![4, 24]);
//!
//! // Or if you just want to test existence, this is faster:
//! assert!(st.contains("quick"));
//! assert!(!st.contains("faux"));
//! ```
//!
//! You can also convert a suffix table to a suffix tree (in linear time):
//!
//! ```rust
//! use suffix::SuffixTable;
//!
//! let st = SuffixTable::new("banana");
//! let tree = st.to_suffix_tree();
//! println!("{:?}", tree);
//! ```
//!
//! There is a command line utility included in this Cargo package called
//! `stree` that will write a suffix tree in GraphViz's `dot` format. From
//! there, it's very easy to visualize it:
//!
//! ```ignore
//! stree "banana" | dot -Tpng > banana.png
//! ```
//!
//! Note that while there are lots of iterators defined for suffix trees in
//! this crate, there is no useful interface for searching text. Namely, suffix
//! tree support is very experimental and my current implementation seems
//! extremely wasteful and not well designed.

#![crate_name = "suffix"]
#![doc(html_root_url = "http://burntsushi.net/rustdoc/suffix")]

#![deny(missing_docs)]

#![allow(unused_features)] // for `test`
#![feature(collections, core, test)]

#[cfg(test)] extern crate quickcheck;
#[cfg(test)] extern crate test;

pub use table::SuffixTable;
pub use tree::{
    Node, SuffixTree,
    Children, Ancestors, Preorder, Leaves, SuffixTreeIndices,
};

// A trivial logging macro. No reason to pull in `log`, which has become
// difficult to use in tests.
macro_rules! lg {
    ($($arg:tt)*) => ({
        let _ = ::std::old_io::stderr().write_str(&*format!($($arg)*));
        let _ = ::std::old_io::stderr().write_str("\n");
    });
}

/// Initialize a `Vec` quickly.
///
/// This is equivalent to `repeat(0).take(len).collect::<Vec<_>>()`.
///
/// TODO: Supposedly, the iterator form should be as fast as this, so we
/// should be able to remove this in the future.
fn vec_from_elem<T: Copy>(len: usize, init: T) -> Vec<T> {
    let mut vec: Vec<T> = Vec::with_capacity(len);
    unsafe { vec.set_len(len); }
    for v in vec.iter_mut() { *v = init; }
    vec
}

/// Binary search to find first element such that `pred(T) == true`.
///
/// Assumes that if `pred(xs[i]) == true` then `pred(xs[i+1]) == true`.
///
/// If all elements yield `pred(T) == false`, then `xs.len()` is returned.
fn binary_search<T, F>(xs: &[T], mut pred: F) -> usize
        where F: FnMut(&T) -> bool {
    let (mut left, mut right) = (0, xs.len());
    while left < right {
        let mid = (left + right) / 2;
        if pred(&xs[mid]) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    left
}

mod array;
mod table;
mod tree;
