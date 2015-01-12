use std::borrow::ToOwned;
use std::cmp::Ordering::{self, Equal, Greater, Less};
use std::collections::hash_map::{HashMap, Entry};
use std::iter::{self, range, repeat};
use std::mem::transmute;
use std::slice;
use std::str;

use SuffixArray;
use self::SuffixType::{Ascending, Descending, Valley};

// const SENTINEL: char = unsafe { transmute(0x110000) };
const SENTINEL: u32 = 0;
const INVALID: usize = 0x110001; // this is wrong

pub fn naive<'s>(s: &'s str) -> SuffixArray<'s> {
    make_suffix_array(s, naive_table(s))
}

pub fn naive_table<'s>(s: &'s str) -> Vec<usize> {
    let mut table: Vec<_> = s.char_indices().map(|(i, _)| i).collect();
    table.sort_by(|&a, &b| s[a..].cmp(&s[b..]));
    table
}

fn make_suffix_array<'s>(s: &'s str, table: Vec<usize>) -> SuffixArray<'s> {
    let mut inverse: Vec<usize> = repeat(0).take(table.len()).collect();
    for (rank, &sufstart) in table.iter().enumerate() {
        inverse[sufstart] = rank;
    }
    let lcp_lens = lcp_lens_linear(s, &*table, &*inverse);
    SuffixArray {
        text: s,
        table: table,
        inverse: inverse,
        lcp_lens: lcp_lens,
    }
}

fn lcp_lens_linear(text: &str, table: &[usize], inv: &[usize]) -> Vec<usize> {
    // This is a linear time construction algorithm taken from the first
    // two slides of:
    // http://www.cs.helsinki.fi/u/tpkarkka/opetus/11s/spa/lecture10.pdf
    //
    // It does require the use of the inverse suffix array, which makes this
    // O(n) in space. The inverse suffix array gives us a special ordering
    // with which to compute the LCPs.
    let mut lcps: Vec<_> = repeat(0).take(table.len()).collect();
    let mut len = 0;
    for (sufi2, &rank) in inv.iter().enumerate() {
        if rank == 0 {
            continue
        }
        let sufi1 = table[rank - 1];
        len += lcp_len(&text[sufi1 + len..], &text[sufi2 + len..]);
        lcps[rank] = len;
        if len > 0 {
            len -= 1;
        }
    }
    lcps
}

fn lcp_lens_quadratic(text: &str, table: &[usize]) -> Vec<usize> {
    // This is quadratic because there are N comparisons for each LCP.
    // But it is done in constant space.

    // The first LCP is always 0 because of the definition:
    //   LCP_LENS[i] = lcp_len(suf[i-1], suf[i])
    let mut lcps: Vec<_> = repeat(0).take(table.len()).collect();
    for (i, win) in table.windows(2).enumerate() {
        lcps[i+1] = lcp_len(&text[win[0]..], &text[win[1]..]);
    }
    lcps
}

/// Compute the length of the least common prefix between two strings.
fn lcp_len(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|&(ca, cb)| ca == cb).count()
}

// pub fn sais<'s>(text: &'s str) -> SuffixArray<'s> {
    // make_suffix_array(text, sais_table(text))
// }

pub fn sais_table<'s>(mut text: String) -> Vec<usize> {
    text.push('\x00');
    // let sufs = sais_vec(Text::from_str(text.as_slice()));
    let sufs = sais_vec(&Unicode::from_str(&*text));
    // sufs.remove(0); // remove the sentinel
    sufs
}

fn sais_vec<T>(text: &T) -> Vec<usize>
        where T: Text,
              <<T as Text>::IdxChars as Iterator>::Item: IdxChar,
              <<T as Text>::Chars as Iterator>::Item: Char {
    println!("finding suffix types");
    let stypes = suffix_types(text);

    // DEBUG.
    debug!("\nsuffix types");
    for (c, t) in text.chars().map(|v| v.char()).zip(stypes.iter()) {
        debug!("{}: {:?}", c, t);
    }
    // DEBUG.

    println!("finding wstrings");
    let wstrs = find_wstrings(&*stypes, text);

    // DEBUG.
    debug!("\nwstrings");
    for (i, wstr) in wstrs.iter().enumerate() {
        debug!("wstr {} [{}, {}): {}",
               i, wstr.start, wstr.end, wstr._string(text));
    }
    // DEBUG.

    println!("sorting {} wstrings", wstrs.len());
    let mut wstrs_sorted = wstrs.clone();
    wstrs_sorted.sort_by(|w1, w2| wstring_cmp(text, &*stypes, w1, w2));

    // DEBUG.
    debug!("\nsorted wstrings");
    for (i, wstr) in wstrs_sorted.iter().enumerate() {
        debug!("wstr {} [{}, {}): {}",
               i, wstr.start, wstr.end, wstr._string(text));
    }
    // DEBUG.

    vec![]
}

// fn sais_vec(text: &str) -> Vec<usize> {
    // println!("finding suffix types");
    // let stypes = suffix_types(chars);
//
    // // DEBUG.
    // debug!("\nsuffix types");
    // for (c, t) in chars.iter().zip(stypes.iter()) {
        // debug!("{}: {}", chr(*c), t);
    // }
    // // DEBUG.
//
    // println!("finding wstrings");
    // let wstrs = find_wstrings(&*stypes, chars);
//
    // // DEBUG.
    // debug!("\nwstrings");
    // for (i, wstr) in wstrs.iter().enumerate() {
        // debug!("wstr {} [{}, {}): {}",
               // i, wstr.start, wstr.end, chrs(wstr.slice(chars)));
    // }
    // // DEBUG.
//
    // println!("sorting {} wstrings", wstrs.len());
    // let mut wstrs_sorted = wstrs.clone();
    // wstrs_sorted.sort_by(|w1, w2| wstring_cmp(&*chars, &*stypes, w1, w2));
//
    // // DEBUG.
    // debug!("\nsorted wstrings");
    // for (i, wstr) in wstrs_sorted.iter().enumerate() {
        // debug!("wstr {} [{}, {}): {}",
               // i, wstr.start, wstr.end, chrs(wstr.slice(chars)));
    // }
    // // DEBUG.
//
    // // Derive lexical names for each wstring. Each wstring gets its own
    // // unique name.
    // println!("building reduced string");
    // let mut duplicate_names = false; // if stays false, we have our base case
    // let mut cur_name = 0;
    // let mut last_wstr = wstrs[0];
    // let mut reduced: Vec<u32> = repeat(0).take(wstrs.len() + 1).collect();
    // reduced[last_wstr.sequence] = cur_name;
    // reduced[wstrs.len()] = SENTINEL;
    // for &wstr in wstrs_sorted.iter().skip(1) {
        // // let order = wstring_cmp(&*chars, &*stypes, &last_wstr, &wstr);
        // // debug!("cmp({}, {}) == {}", last_wstr, wstr, order);
        // if wstring_cmp(&*chars, &*stypes, &last_wstr, &wstr) == Equal {
            // duplicate_names = true;
        // } else {
            // cur_name += 1;
            // // never allow a name to be our sentinel
            // if cur_name == SENTINEL {
                // cur_name += 1;
            // }
        // }
        // reduced[wstr.sequence] = cur_name;
        // last_wstr = wstr;
    // }
//
    // // DEBUG.
    // debug!("reduced string: {}", reduced);
    // // DEBUG.
//
    // if duplicate_names {
        // println!("size of recursive case: {}, names: {}", reduced.len(), cur_name);
        // let sa = sais_vec(&*reduced);
        // // let mut sa: Vec<usize> = range(0, reduced.len()).collect();
        // // sa.sort_by(|&a, &b| reduced[a..].cmp(reduced[b..]));
        // // debug!("SA: {}", sa);
        // // Drop the first suffix because it is always the sentinel.
        // for (rank, &sufstart) in sa.iter().skip(1).enumerate() {
            // wstrs_sorted[rank] = wstrs[sufstart];
        // }
//
        // // DEBUG.
        // debug!("\nreduced suffixes");
        // for (i, &sufstart) in sa.iter().enumerate() {
            // debug!("{}: [{}..]: {}", i, sufstart, reduced[sufstart..]);
        // }
        // // DEBUG.
    // }
//
    // // DEBUG.
    // debug!("\nsorted wstrings");
    // for (i, wstr) in wstrs_sorted.iter().enumerate() {
        // debug!("wstr {} [{}, {}): {}",
               // i, wstr.start, wstr.end, chrs(wstr.slice(chars)));
    // }
    // // DEBUG.
//
    // // TODO: This nonsense could probably be replaced with a BTreeMap.
    // // But, we still have sentinels, which tweak the ordering of characters.
    // // If we get rid of sentinels, a BTreeMap is trivial to use.
    // // Otherwise, we need to newtype a character and define an ordering on it.
    // let mut sa: Vec<int> = repeat(-1).take(chars.len()).collect();
    // let mut bin_sizes: HashMap<u32, usize> = HashMap::new();
    // for &c in chars.iter() {
        // match bin_sizes.entry(&c) {
            // Entry::Vacant(v) => { v.insert(1); }
            // Entry::Occupied(mut v) => { *v.get_mut() += 1; }
        // }
    // }
//
    // // Find the alphas in sorted order. These correspond to the labels of
    // // each bin.
    // let mut alphas: Vec<u32> = bin_sizes.keys().map(|&c| c).collect();
    // alphas.sort_by(chrcmp);
//
    // // These are pointers to the start/end of each bin. They are regenerated
    // // at each step.
    // let mut bin_ptrs: HashMap<u32, usize> = HashMap::new();
//
    // // Find the index of the last element of each bin in `sa`.
    // let mut sum = 0u;
    // for &c in alphas.iter() {
        // sum += bin_sizes[c];
        // bin_ptrs.insert(c, sum - 1);
    // }
//
    // // Insert the valley suffixes.
    // for wstr in wstrs_sorted.iter().rev() {
        // let binp = &mut bin_ptrs[chars[wstr.start]];
        // sa[*binp] = wstr.start as int;
        // if *binp > 0 { *binp -= 1; }
    // }
    // debug!("after step 0: {}", sa);
//
    // // Now find the start of each bin.
    // let mut sum = 0u;
    // for &c in alphas.iter() {
        // bin_ptrs.insert(c, sum);
        // sum += bin_sizes[c];
    // }
//
    // // Insert the descending suffixes.
    // for i in range(0, sa.len()) {
        // let sufi = sa[i];
        // if sufi > 0 && stypes[(sufi - 1) as usize].is_desc() {
            // let binp = &mut bin_ptrs[chars[(sufi - 1) as usize]];
            // sa[*binp] = sufi - 1;
            // *binp += 1;
        // }
    // }
    // debug!("after step 1: {}", sa);
//
    // // ... and find the end of each bin again.
    // let mut sum = 0u;
    // for &c in alphas.iter() {
        // sum += bin_sizes[c];
        // bin_ptrs.insert(c, sum - 1);
    // }
//
    // // Insert the ascending suffixes.
    // for i in range(0, sa.len()).rev() {
        // let sufi = sa[i];
        // if sufi > 0 && stypes[(sufi - 1) as usize].is_asc() {
            // let binp = &mut bin_ptrs[chars[(sufi - 1) as usize]];
            // sa[*binp] = sufi - 1;
            // *binp -= 1;
        // }
    // }
    // debug!("after step 2: {}", sa);
    // unsafe { ::std::mem::transmute(sa) }
// }

#[derive(Clone, Copy, Eq, Show)]
enum SuffixType {
    Ascending,
    Descending,
    Valley,
}

impl SuffixType {
    fn is_asc(&self) -> bool {
        match *self {
            Ascending | Valley => true,
            _ => false,
        }
    }

    fn is_desc(&self) -> bool {
        if let Descending = *self { true } else { false }
    }

    fn is_valley(&self) -> bool {
        if let Valley = *self { true } else { false }
    }

    fn inherit(&self) -> SuffixType {
        match *self {
            Valley => Ascending,
            _ => *self,
        }
    }
}

impl PartialEq for SuffixType {
    fn eq(&self, other: &SuffixType) -> bool {
        (self.is_asc() && other.is_asc())
        || (self.is_desc() && other.is_desc())
    }
}

impl PartialOrd for SuffixType {
    fn partial_cmp(&self, other: &SuffixType) -> Option<Ordering> {
        Some(if self.is_desc() && other.is_asc() {
            Less
        } else if self.is_asc() && other.is_desc() {
            Greater
        } else {
            assert!(self == other);
            Equal
        })
    }
}

impl Ord for SuffixType {
    fn cmp(&self, other: &SuffixType) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

fn suffix_types<T>(text: &T) -> Vec<SuffixType>
        where T: Text, <<T as Text>::IdxChars as Iterator>::Item: IdxChar {
    let mut stypes = Vec::with_capacity(text.len());
    let (mut last, mut lastc) = (Valley, SENTINEL);
    for (i, c) in text.char_indices().rev().map(|v| v.idx_char()) {
        if c == SENTINEL {
            stypes.push(Valley);
            lastc = c;
            continue;
        }
        if c < lastc {
            stypes.push(Ascending);
        } else if c > lastc {
            stypes.push(Descending);
        } else {
            stypes.push(last.inherit());
        }
        let len = stypes.len();
        if stypes[len-1].is_desc() && stypes[len-2].is_asc() {
            stypes[len-2] = Valley;
        }
        last = stypes[len-1];
        lastc = c;
    }
    stypes.reverse();
    stypes
}

#[derive(Copy, Clone, Show)]
struct WString {
    sequence: usize,
    start: usize,
    end: usize,
}

impl WString {
    fn start_at(sequence: usize, i: usize) -> WString {
        WString { sequence: sequence, start: i, end: 0 }
    }

    fn index<'s, T>(&self, xs: &'s [T], i: usize) -> &'s T {
        assert!(self.start + i < self.end);
        &xs[i]
    }

    fn char_at<T: Text>(&self, text: &T, i: usize) -> u32 {
        assert!(self.start + i < self.end);
        text.char_at(self.start + i)
    }

    fn len(&self) -> usize {
        self.end - self.start
    }

    // For debugging only.
    fn _string<T: Text>(&self, text: &T) -> String {
        text._string(self.start, self.end)
    }
}

fn find_wstrings<T>(stypes: &[SuffixType], text: &T) -> Vec<WString>
        where T: Text, <<T as Text>::IdxChars as Iterator>::Item: IdxChar {
    let mut wstrs: Vec<WString> = Vec::with_capacity(text.len() / 4);
    let mut sequence = 0;
    let mut cur = WString::start_at(sequence, 0);

    for ((bytei, c), t) in text.char_indices().map(|v| v.idx_char()).zip(stypes.iter()) {
        if t.is_valley() {
            if cur.start == 0 {
                cur.start = bytei;
            } else {
                cur.end = bytei + 1;
                wstrs.push(cur);
                sequence += 1;
                cur = WString::start_at(sequence, bytei);
            }
        }
    }
    // We might need to do something different here if we don't have
    // a sentinel.
    // Although... Since we've converted everything to `u32`, we could just
    // pick an arbitrary `u32` that is not a valid Unicode scalar value. ---AG
    cur.end = text.len();
    wstrs.push(cur);
    wstrs
}

fn wstring_cmp<T: Text>(
    text: &T,
    stypes: &[SuffixType],
    w1: &WString,
    w2: &WString,
) -> Ordering {
    for i in range(0, w1.len()) {
        // If we've run out of characters in w2, then w2 is a proper prefix
        // of w1 (since we still have more characters in w1 to examine).
        // Therefore, w1 > w2.
        if i >= w2.len() {
            return Greater;
        }
        match w1.char_at(text, i).cmp(&w2.char_at(text, i)) {
            Equal => {
                match w1.index(stypes, i).cmp(w2.index(stypes, i)) {
                    Equal => { continue }
                    ord => return ord,
                }
            }
            ord => return ord,
        }
    }
    // At this point, we know that w1 is some prefix of w2.
    // If it's a proper prefix, then w1 < w2. Otherwise, they are equal.
    if w1.len() < w2.len() {
        Less
    } else {
        Equal
    }
}

fn chrs(ns: &[u32]) -> String {
    ns.iter().map(|&n| chr(n)).collect()
}

fn chr(n: u32) -> char {
    unreachable!()
    // if n == SENTINEL {
        // '$'
    // } else if n == INVALID as u32 { // wrong
        // '?'
    // } else {
        // ::std::char::from_u32(n).unwrap()
    // }
}

#[inline]
fn chrcmp(c1: &u32, c2: &u32) -> Ordering {
    unreachable!()
    // if *c1 == SENTINEL && *c2 != SENTINEL {
        // Less
    // } else if *c1 != SENTINEL && *c2 == SENTINEL {
        // Greater
    // } else {
        // c1.cmp(c2)
    // }
}

trait Text {
    type Chars: Iterator + DoubleEndedIterator;
    type IdxChars: Iterator + DoubleEndedIterator;
    fn len(&self) -> usize;
    fn char_at(&self, i: usize) -> u32;
    fn chars(&self) -> Self::Chars;
    fn char_indices(&self) -> Self::IdxChars;

    // For debugging. (This is why we aren't implementing slice syntax.)
    fn _string_from(&self, start: usize) -> String;
    fn _string(&self, start: usize, end: usize) -> String;
}

struct Unicode<'s> {
    s: &'s str,
    len: usize,
}

impl<'s> Unicode<'s> {
    fn from_str(s: &'s str) -> Unicode<'s> {
        Unicode::from_str_len(s, s.chars().count())
    }

    fn from_str_len(s: &'s str, len: usize) -> Unicode<'s> {
        Unicode { s: s, len: len }
    }
}

impl<'s> Text for Unicode<'s> {
    type Chars = str::Chars<'s>;
    type IdxChars = str::CharIndices<'s>;

    fn len(&self) -> usize { self.len }

    fn char_at(&self, i: usize) -> u32 { self.s.char_at(i) as u32 }

    fn chars(&self) -> str::Chars<'s> {
        self.s.chars()
    }

    fn char_indices(&self) -> str::CharIndices<'s> {
        self.s.char_indices()
    }

    fn _string_from(&self, start: usize) -> String {
        self.s[start..].to_owned()
    }

    fn _string(&self, start: usize, end: usize) -> String {
        self.s[start..end].to_owned()
    }
}

struct LexNames<'s> {
    s: &'s [u32],
}

impl<'s> Text for LexNames<'s> {
    type Chars = slice::Iter<'s, u32>;
    type IdxChars = iter::Enumerate<slice::Iter<'s, u32>>;

    fn len(&self) -> usize { self.s.len() }

    fn char_at(&self, i: usize) -> u32 { self.s[i] }

    fn chars(&self) -> slice::Iter<'s, u32> {
        self.s.iter()
    }

    fn char_indices(&self) -> iter::Enumerate<slice::Iter<'s, u32>> {
        self.s.iter().enumerate()
    }

    fn _string_from(&self, start: usize) -> String {
        format!("{:?}", &self.s[start..])
    }

    fn _string(&self, start: usize, end: usize) -> String {
        format!("{:?}", &self.s[start..end])
    }
}

trait IdxChar {
    fn idx_char(self) -> (usize, u32);
}

impl<'a> IdxChar for (usize, &'a u32) {
    #[inline]
    fn idx_char(self) -> (usize, u32) { (self.0, *self.1) }
}

impl IdxChar for (usize, char) {
    #[inline]
    fn idx_char(self) -> (usize, u32) { (self.0, self.1 as u32) }
}

trait Char {
    fn char(self) -> u32;
}

impl<'a> Char for &'a u32 {
    #[inline]
    fn char(self) -> u32 { *self }
}

impl Char for char {
    #[inline]
    fn char(self) -> u32 { self as u32 }
}

// enum Text<'s> {
    // Unicode(&'s str, usize),
    // LexNames(Vec<u32>),
// }
//
// impl<'s> Text<'s> {
    // fn from_str(s: &'s str) -> Text<'s> {
        // Text::from_str_len(s, s.chars().count())
    // }
//
    // fn from_str_len(s: &'s str, len: usize) -> Text<'s> {
        // Text::Unicode(s, len)
    // }
//
    // fn from_names(names: Vec<u32>) -> Text<'static> {
        // Text::LexNames(names)
    // }
//
    // fn len(&self) -> usize {
        // match *self {
            // Text::Unicode(_, len) => len,
            // Text::LexNames(ref names) => names.len(),
        // }
    // }
//
    // fn char_at(&self, i: usize) -> u32 {
        // match *self {
            // Text::Unicode(s, _) => s.char_at(i) as u32,
            // Text::LexNames(ref s) => s[i],
        // }
    // }
//
    // // Should be for debugging only!
    // fn string_from(&self, i: usize) -> String {
        // match *self {
            // Text::Unicode(s, _) => s[i..].to_owned(),
            // Text::LexNames(ref s) => format!("{:?}", s),
        // }
    // }
// }

#[cfg(test)]
mod tests {
    extern crate test;

    use self::test::Bencher;
    use quickcheck::{TestResult, quickcheck};
    use super::{naive, naive_table, sais_table};

    // #[test]
    // fn sais_basic1() {
        // assert_eq!(naive("apple"), sais("apple"));
    // }
//
    // #[test]
    // fn sais_basic2() {
        // assert_eq!(naive("banana"), sais("banana"));
    // }
//
    // #[test]
    // fn sais_basic3() {
        // assert_eq!(naive("mississippi"), sais("mississippi"));
    // }
//
    // #[test]
    // fn qc_naive_equals_sais() {
        // fn prop(s: String) -> TestResult {
            // if s.is_empty() { return TestResult::discard(); }
            // TestResult::from_bool(naive(&*s) == sais(&*s))
        // }
        // quickcheck(prop as fn(String) -> TestResult);
    // }

    #[test]
    fn array_scratch() {
        let s = "tgtgtgtgcaccg";
        let sa = sais_table(s.to_string());
        // let sa = sais("32P32Pz");

        debug!("\n\ngot suffix array:");
        for &i in sa.iter() {
            debug!("{:2}: {}", i, &s[i..]);
        }
        debug!("\n\nnaive suffix array:");
        for &i in naive_table(&*s).iter() {
            debug!("{:2}: {}", i, &s[i..]);
        }

        // assert_eq!(sa, naive("32P32Pz"));
    }

    // #[bench]
    // fn naive_small(b: &mut Bencher) {
        // let s = "mississippi";
        // b.iter(|| { naive_table(s); })
    // }
//
    // #[bench]
    // fn sais_small(b: &mut Bencher) {
        // let s = "mississippi";
        // b.iter(|| { sais_table(s); })
    // }
//
    // #[bench]
    // fn naive_dna_small(b: &mut Bencher) {
        // let s = include_str!("AP009048_10000.fasta");
        // b.iter(|| { naive_table(s); })
    // }
//
    // #[bench]
    // fn sais_dna_small(b: &mut Bencher) {
        // let s = include_str!("AP009048_10000.fasta");
        // b.iter(|| { sais_table(s); })
    // }
//
    // #[bench]
    // fn naive_dna_medium(b: &mut Bencher) {
        // let s = include_str!("AP009048_100000.fasta");
        // b.iter(|| { naive_table(s); })
    // }
//
    // #[bench]
    // fn sais_dna_medium(b: &mut Bencher) {
        // let s = include_str!("AP009048_100000.fasta");
        // b.iter(|| { sais_table(s); })
    // }
}
