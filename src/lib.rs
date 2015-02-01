//! Documentation for the `suffix` crate.

#![crate_name = "suffix"]
#![doc(html_root_url = "http://burntsushi.net/rustdoc/suffix")]

// #![allow(dead_code, unused_imports, unused_variables)]
#![allow(unused_features)] // for `test`
#![feature(collections, core, test)]

#[cfg(test)] extern crate quickcheck;
#[cfg(test)] extern crate test;

pub use array::SuffixArray;
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
