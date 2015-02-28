#![feature(old_io, test)]

extern crate quickcheck;
extern crate suffix;
extern crate test;

use quickcheck::{QuickCheck, TestResult};
use suffix::SuffixTable;

// A trivial logging macro. No reason to pull in `log`, which has become
// difficult to use in tests.
macro_rules! lg {
    ($($arg:tt)*) => ({
        let _ = writeln!(&mut ::std::old_io::stderr(), $($arg)*);
    });
}

fn sais(text: &str) -> SuffixTable { SuffixTable::new(text) }
fn naive(text: &str) -> SuffixTable { SuffixTable::new_naive(text) }

// These tests assume the correctness of the `naive` method of computing a
// suffix array. (It's only a couple lines of code and probably difficult to
// get wrong.)

#[test]
fn basic1() {
    assert_eq!(naive("apple"), sais("apple"));
}

#[test]
fn basic2() {
    assert_eq!(naive("banana"), sais("banana"));
}

#[test]
fn basic3() {
    assert_eq!(naive("mississippi"), sais("mississippi"));
}

#[test]
fn basic4() {
    assert_eq!(naive("tgtgtgtgcaccg"), sais("tgtgtgtgcaccg"));
}

#[test]
fn empty_is_ok() {
    assert_eq!(naive(""), sais(""));
}

#[test]
fn one_is_ok() {
    assert_eq!(naive("a"), sais("a"));
}

#[test]
fn two_diff_is_ok() {
    assert_eq!(naive("ab"), sais("ab"));
}

#[test]
fn two_same_is_ok() {
    assert_eq!(naive("aa"), sais("aa"));
}

#[test]
fn qc_naive_equals_sais() {
    fn prop(s: String) -> TestResult {
        if s.is_empty() { return TestResult::discard(); }
        let expected = naive(&*s);
        let got = sais(&*s);
        TestResult::from_bool(expected == got)
    }
    QuickCheck::new()
        .tests(1000)
        .max_tests(50000)
        .quickcheck(prop as fn(String) -> TestResult);
}

// Do some testing on substring search.

#[test]
fn empty_find_empty() {
    let sa = sais("");
    assert_eq!(sa.positions(""), vec![]);
    assert!(!sa.contains(""));
}

#[test]
fn empty_find_one() {
    let sa = sais("");
    assert_eq!(sa.positions("a"), vec![]);
    assert!(!sa.contains("a"));
}

#[test]
fn empty_find_two() {
    let sa = sais("");
    assert_eq!(sa.positions("ab"), vec![]);
    assert!(!sa.contains("ab"));
}

#[test]
fn one_find_empty() {
    let sa = sais("a");
    assert_eq!(sa.positions(""), vec![]);
    assert!(!sa.contains(""));
}

#[test]
fn one_find_one_notexists() {
    let sa = sais("a");
    assert_eq!(sa.positions("b"), vec![]);
    assert!(!sa.contains("b"));
}

#[test]
fn one_find_one_exists() {
    let sa = sais("a");
    assert_eq!(sa.positions("a"), vec![0]);
    assert!(sa.contains("a"));
}

#[test]
fn two_find_one_exists() {
    let sa = sais("ab");
    assert_eq!(sa.positions("b"), vec![1]);
    assert!(sa.contains("b"));
}

#[test]
fn two_find_two_exists() {
    let sa = sais("aa");
    assert_eq!(sa.positions("a"), vec![1, 0]);
    assert!(sa.contains("a"));
}

#[test]
fn many_exists() {
    let sa = sais("zzzzzaazzzzz");
    assert_eq!(sa.positions("a"), vec![5, 6]);
    assert!(sa.contains("a"));
}

#[test]
fn many_exists_long() {
    let sa = sais("zzzzabczzzzzabczzzzzz");
    assert_eq!(sa.positions("abc"), vec![4, 12]);
    assert!(sa.contains("abc"));
}

#[test]
fn query_longer() {
    let sa = sais("az");
    assert_eq!(sa.positions("mnomnomnomnomnomnomno"), vec![]);
    assert!(!sa.contains("mnomnomnomnomnomnomno"));
}

#[test]
fn query_longer_less() {
    let sa = sais("zz");
    assert_eq!(sa.positions("mnomnomnomnomnomnomno"), vec![]);
    assert!(!sa.contains("mnomnomnomnomnomnomno"));
}

#[test]
fn query_longer_greater() {
    let sa = sais("aa");
    assert_eq!(sa.positions("mnomnomnomnomnomnomno"), vec![]);
    assert!(!sa.contains("mnomnomnomnomnomnomno"));
}

#[test]
fn query_spaces() {
    let sa = sais("The quick brown fox was very quick.");
    assert_eq!(sa.positions("quick"), vec![4, 29]);
}

#[test]
fn unicode_snowman() {
    let sa = sais("☃abc☃");
    assert!(sa.contains("☃"));
    assert_eq!(sa.positions("☃"), vec![6, 0]);
}
