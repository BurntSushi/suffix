//! This crate provides fast suffix array construction with Unicode support.
//!
//! The details of the construction algorithm are documented on the
//! `SuffixTable` type.
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
//! assert_eq!(st.positions("quick"), &[4, 24]);
//!
//! // Or if you just want to test existence, this is faster:
//! assert!(st.contains("quick"));
//! assert!(!st.contains("faux"));
//! ```

#![crate_name = "suffix"]
#![doc(html_root_url = "http://burntsushi.net/rustdoc/suffix")]

#![deny(missing_docs)]

#[cfg(test)] extern crate quickcheck;

pub use table::SuffixTable;

// A trivial logging macro. No reason to pull in `log`, which has become
// difficult to use in tests.
macro_rules! lg {
    ($($arg:tt)*) => ({
        use std::io::{Write, stderr};
        writeln!(&mut stderr(), $($arg)*).unwrap();
    });
}

mod table;
