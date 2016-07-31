extern crate quickcheck;
extern crate suffix;

use quickcheck::{QuickCheck, TestResult, Testable};
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

fn qc<T: Testable>(f: T) {
    QuickCheck::new().tests(1000).max_tests(10000).quickcheck(f);
}

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
fn nul_is_ok() {
    assert_eq!(naive("\x00"), sais("\x00"));
}

#[test]
fn snowman_is_ok() {
    assert_eq!(naive("☃abc☃"), sais("☃abc☃"));
}

// See if we can catch any corner cases we forgot about.
#[test]
fn prop_naive_equals_sais() {
    fn prop(s: String) -> TestResult {
        if s.is_empty() { return TestResult::discard(); }
        let expected = naive(&*s);
        let got = sais(&*s);
        TestResult::from_bool(expected == got)
    }
    qc(prop as fn(String) -> TestResult);
}

#[test]
fn prop_matches_naive() {
    fn prop(s: String) -> bool {
        let expected_table = SuffixTable::new_naive(&*s);
        let expected = expected_table.table();
        let got_table = SuffixTable::new(&*s);
        let got = got_table.table();
        expected == got
    }
    qc(prop as fn(String) -> bool);
}

// Do some testing on substring search.

#[test]
fn empty_find_empty() {
    let sa = sais("");
    assert_eq!(sa.positions(""), &[]);
    assert!(!sa.contains(""));
}

#[test]
fn empty_find_one() {
    let sa = sais("");
    assert_eq!(sa.positions("a"), &[]);
    assert!(!sa.contains("a"));
}

#[test]
fn empty_find_two() {
    let sa = sais("");
    assert_eq!(sa.positions("ab"), &[]);
    assert!(!sa.contains("ab"));
}

#[test]
fn one_find_empty() {
    let sa = sais("a");
    assert_eq!(sa.positions(""), &[]);
    assert!(!sa.contains(""));
}

#[test]
fn one_find_one_notexists() {
    let sa = sais("a");
    assert_eq!(sa.positions("b"), &[]);
    assert!(!sa.contains("b"));
}

#[test]
fn one_find_one_exists() {
    let sa = sais("a");
    assert_eq!(sa.positions("a"), &[0]);
    assert!(sa.contains("a"));
}

#[test]
fn two_find_one_exists() {
    let sa = sais("ab");
    assert_eq!(sa.positions("b"), &[1]);
    assert!(sa.contains("b"));
}

#[test]
fn two_find_two_exists() {
    let sa = sais("aa");
    assert_eq!(vec![1, 0], sa.positions("a"));
    assert!(sa.contains("a"));
}

#[test]
fn many_exists() {
    let sa = sais("zzzzzaazzzzz");
    assert_eq!(vec![5, 6], sa.positions("a"));
    assert!(sa.contains("a"));
}

#[test]
fn many_exists_long() {
    let sa = sais("zzzzabczzzzzabczzzzzz");
    assert_eq!(sa.positions("abc"), &[4, 12]);
    assert!(sa.contains("abc"));
}

#[test]
fn parts() {
    let sa = sais("poëzie");
    let sa2 = sa.clone();

    let (data, table) = sa2.into_parts();
    let sa3 = SuffixTable::from_parts(data, table);

    assert_eq!(sa, sa3);
}

#[test]
fn query_longer() {
    let sa = sais("az");
    assert_eq!(sa.positions("mnomnomnomnomnomnomno"), &[]);
    assert!(!sa.contains("mnomnomnomnomnomnomno"));
}

#[test]
fn query_longer_less() {
    let sa = sais("zz");
    assert_eq!(sa.positions("mnomnomnomnomnomnomno"), &[]);
    assert!(!sa.contains("mnomnomnomnomnomnomno"));
}

#[test]
fn query_longer_greater() {
    let sa = sais("aa");
    assert_eq!(sa.positions("mnomnomnomnomnomnomno"), &[]);
    assert!(!sa.contains("mnomnomnomnomnomnomno"));
}

#[test]
fn query_spaces() {
    let sa = sais("The quick brown fox was very quick.");
    assert_eq!(sa.positions("quick"), &[4, 29]);
}

#[test]
fn unicode_snowman() {
    let sa = sais("☃abc☃");
    assert!(sa.contains("☃"));
    assert_eq!(sa.positions("☃"), &[6, 0]);
}

#[test]
fn prop_length() {
    fn prop(s: String) -> bool {
        s.len() == sais(&s).len()
    }
    qc(prop as fn(String) -> bool);
}

#[test]
fn prop_contains() {
    fn prop(s: String, c: u8) -> bool {
        let c = (c as char).to_string();
        s.contains(&c) == sais(&s).contains(&c)
    }
    qc(prop as fn(String, u8) -> bool);
}

#[test]
fn prop_positions() {
    fn prop(s: String, c: u8) -> bool {
        let c = (c as char).to_string();
        let expected: Vec<_> =
            s.match_indices(&c).map(|(s, _)| s as u32).collect();
        let mut got = sais(&s).positions(&c).to_vec();
        got.sort();
        expected == got
    }
    qc(prop as fn(String, u8) -> bool);
}
