use std::borrow::ToOwned;
use std::cmp::Ordering::{self, Equal, Greater, Less};
use std::collections::btree_map::{BTreeMap, Entry};
use std::iter::{self, repeat};
use std::mem::transmute;
use std::num::Int;
use std::slice;
use std::str::{self, CharRange};
use std::u32;

use SuffixArray;
use self::SuffixType::{Ascending, Descending, Valley};

pub fn naive<'s>(s: &'s str) -> SuffixArray<'s> {
    make_suffix_array(s, naive_table(s))
}

pub fn naive_table<'s>(s: &'s str) -> Vec<u32> {
    let mut table: Vec<_> = s.char_indices().map(|(i, _)| i as u32).collect();
    table.sort_by(|&a, &b| s[a as usize..].cmp(&s[b as usize..]));
    table
}

fn make_suffix_array<'s>(s: &'s str, table: Vec<u32>) -> SuffixArray<'s> {
    let mut inverse: Vec<u32> = repeat(0).take(table.len()).collect();
    for (rank, &sufstart) in table.iter().enumerate() {
        inverse[sufstart as usize] = rank as u32;
    }
    let lcp_lens = lcp_lens_linear(s, &*table, &*inverse);
    SuffixArray {
        text: s,
        table: table,
        inverse: inverse,
        lcp_lens: lcp_lens,
    }
}

fn lcp_lens_linear(text: &str, table: &[u32], inv: &[u32]) -> Vec<u32> {
    // This is a linear time construction algorithm taken from the first
    // two slides of:
    // http://www.cs.helsinki.fi/u/tpkarkka/opetus/11s/spa/lecture10.pdf
    //
    // It does require the use of the inverse suffix array, which makes this
    // O(n) in space. The inverse suffix array gives us a special ordering
    // with which to compute the LCPs.
    let mut lcps: Vec<_> = repeat(0).take(table.len()).collect();
    let mut len = 0u32;
    for (sufi2, &rank) in inv.iter().enumerate() {
        if rank == 0 {
            continue
        }
        let sufi1 = table[(rank - 1) as usize];
        len += lcp_len(&text[(sufi1 + len) as usize..],
                       &text[(sufi2 as u32 + len) as usize..]);
        lcps[rank as usize] = len;
        if len > 0 {
            len -= 1;
        }
    }
    lcps
}

fn lcp_lens_quadratic(text: &str, table: &[u32]) -> Vec<u32> {
    // This is quadratic because there are N comparisons for each LCP.
    // But it is done in constant space.

    // The first LCP is always 0 because of the definition:
    //   LCP_LENS[i] = lcp_len(suf[i-1], suf[i])
    let mut lcps: Vec<_> = repeat(0).take(table.len()).collect();
    for (i, win) in table.windows(2).enumerate() {
        lcps[i+1] = lcp_len(&text[win[0] as usize..],
                            &text[win[1] as usize..]);
    }
    lcps
}

/// Compute the length of the least common prefix between two strings.
fn lcp_len(a: &str, b: &str) -> u32 {
    a.chars().zip(b.chars()).take_while(|&(ca, cb)| ca == cb).count() as u32
}

pub fn sais<'s>(text: &'s str) -> SuffixArray<'s> {
    make_suffix_array(text, sais_table(text))
}

pub fn sais_table<'s>(text: &'s str) -> Vec<u32> {
    match text.len() {
        0 => return vec![],
        1 => return vec![0],
        _ => {},
    }

    let chars = text.chars().count();
    println!("Allocating suffix array of size {:?}", chars);
    let mut sa: Vec<u32> = Vec::with_capacity(chars);
    unsafe { sa.set_len(chars); }
    for i in (0..chars) { sa[i] = 0; }
    // let mut sa: Vec<u32> = repeat(0).take(chars).collect();
    println!("done allocating suffix array");

    let mut stypes = SuffixTypes::new(text.len() as u32);
    let mut bins = Bins::new();

    sais_vec(&mut *sa, &mut stypes, &mut bins, &Unicode::from_str(text));
    sa
}

fn sais_vec<T>(sa: &mut [u32], stypes: &mut SuffixTypes,
               bins: &mut Bins, text: &T)
        where T: Text,
              <<T as Text>::IdxChars as Iterator>::Item: IdxChar,
              <<T as Text>::Chars as Iterator>::Item: Char {
    match text.len() {
        0 => return,
        1 => { sa[0] = 0; return; }
        _ => {},
    }

    for v in sa.iter_mut() { *v = 0; }

    stypes.compute(text);
    bins.find_sizes(text.chars().map(|c| c.char()));
    bins.find_tail_pointers();

    // Insert the valley suffixes.
    for (i, c) in text.char_indices().map(|v| v.idx_char()) {
        if stypes.is_valley(i as u32) {
            bins.tail_insert(sa, i as u32, c);
        }
    }
    debug!("{{wstr}} after step 0: {:?}", sa);
    println!("wstr step 0 complete");

    // Now find the start of each bin.
    bins.find_head_pointers();
    println!("wstr step 1 half complete");

    // Insert the descending suffixes.
    let (lasti, lastc) = text.prev(text.len());
    if stypes.is_desc(lasti) {
        bins.head_insert(sa, lasti, lastc);
    }
    for i in 0..sa.len() {
        let sufi = sa[i];
        if sufi > 0 {
            let (lasti, lastc) = text.prev(sufi);
            if stypes.is_desc(lasti) {
                bins.head_insert(sa, lasti, lastc);
            }
        }
    }

    debug!("{{wstr}} after step 1: {:?}", sa);
    println!("wstr step 1 complete");

    bins.find_tail_pointers();

    // Insert the ascending suffixes.
    for i in (0..sa.len()).rev() {
        let sufi = sa[i];
        if sufi > 0 {
            let (lasti, lastc) = text.prev(sufi);
            if stypes.is_asc(lasti) {
                bins.tail_insert(sa, lasti, lastc);
            }
        }
    }

    debug!("{{wstr}} after step 2: {:?}", sa);
    println!("wstr step 2 complete");

    let mut num_wstrs = 0u32;
    for i in 0..sa.len() {
        let sufi = sa[i];
        if stypes.is_valley(sufi) {
            sa[num_wstrs as usize] = sufi;
            num_wstrs += 1;
        }
    }
    // This check is necessary because we don't have a sentinel, which would
    // normally guarantee at least one wstring.
    if num_wstrs == 0 { num_wstrs = 1; }

    let mut prev_sufi = 0u32; // the first suffix can never be a valley
    let mut name = 0u32;
    // We set our "name buffer" to be max u32 values. Since there are at
    // most n/2 wstrings, a name can never be greater than n/2.
    for i in (num_wstrs..(sa.len() as u32)) { sa[i as usize] = u32::MAX; }
    for i in (0..num_wstrs) {
        let this_sufi = sa[i as usize];
        let mut diff = false;
        let mut thisi = this_sufi;
        let mut previ = prev_sufi;
        loop {
            if thisi >= text.len() || previ >= text.len() {
                // This means the next comparison *should* hit the sentinel,
                // but we don't store the sentinel. The sentinel must never be
                // equal to any other character, so we have a diff!
                diff = true;
                break;
            }
            let this_char = text.char_at(thisi);
            let prev_char = text.char_at(previ);
            if prev_sufi == 0
                    || this_char != prev_char
                    || !stypes.equal(thisi, previ) {
                diff = true;
                break;
            }
            if thisi > this_sufi && (stypes.is_valley(thisi)
                                     || stypes.is_valley(previ)) {
                break;
            }
            thisi = text.next(thisi).0;
            previ = text.next(previ).0;
        }
        if diff {
            name += 1;
            prev_sufi = this_sufi;
        }
        // This divide-by-2 trick only works because it's impossible to have
        // two wstrings start at adjacent locations (they must at least be
        // separated by a single descending character).
        sa[(num_wstrs + (this_sufi / 2)) as usize] = name - 1;
    }

    println!("{:?} lexical names found", name);
    debug!("{:?} lexical names found", name);

    // We've inserted the lexical names into the latter half of the suffix
    // array, but it's sparse. so let's smush them all up to the end.
    let mut j = sa.len() as u32 - 1;
    for i in (num_wstrs..(sa.len() as u32)).rev() {
        if sa[i as usize] != u32::MAX {
            sa[j as usize] = sa[i as usize];
            j -= 1;
        }
    }

    println!("built reduced string (size: {:?})", num_wstrs);
    debug!("reduced: {:?}", &sa[sa.len() - (num_wstrs as usize) ..]);

    if name < num_wstrs {
        let split_at = sa.len() - (num_wstrs as usize);
        let (r_sa, r_text) = sa.split_at_mut(split_at);
        sais_vec(r_sa.slice_to_mut(num_wstrs as usize), stypes, bins,
                 &LexNames(r_text));
        stypes.compute(text);
    } else {
        for i in (0..num_wstrs) {
            let reducedi = sa[((sa.len() as u32) - num_wstrs + i) as usize];
            sa[reducedi as usize] = i;
        }
    }

    debug!("reduced sa: {:?}", sa);
    println!("SA of reduced string computed");

    let mut j = sa.len() as u32 - num_wstrs;
    for (i, c) in text.char_indices().map(|v| v.idx_char()) {
        if stypes.is_valley(i as u32) {
            sa[j as usize] = i as u32;
            j += 1;
        }
    }
    for i in (0..num_wstrs) {
        let sufi = sa[i as usize];
        sa[i as usize] = sa[(sa.len() as u32 - num_wstrs + sufi) as usize];
    }
    for i in (num_wstrs..(sa.len() as u32)) {
        sa[i as usize] = 0;
    }

    bins.find_sizes(text.chars().map(|c| c.char()));
    bins.find_tail_pointers();

    // Insert the valley suffixes.
    for i in (0..num_wstrs).rev() {
        let sufi = sa[i as usize];
        sa[i as usize] = 0;
        bins.tail_insert(sa, sufi, text.char_at(sufi));
    }
    debug!("after step 0: {:?}", sa);
    println!("step 0 complete (added {:?} suffixes)", num_wstrs);

    // Now find the start of each bin.
    bins.find_head_pointers();
    println!("step 1 half complete");

    // Insert the descending suffixes.
    let (lasti, lastc) = text.prev(text.len());
    if stypes.is_desc(lasti) {
        bins.head_insert(sa, lasti, lastc);
    }
    for i in 0..sa.len() {
        let sufi = sa[i];
        if sufi > 0 {
            let (lasti, lastc) = text.prev(sufi);
            if stypes.is_desc(lasti) {
                bins.head_insert(sa, lasti, lastc);
            }
        }
    }
    debug!("after step 1: {:?}", sa);
    println!("step 1 complete");

    // ... and find the end of each bin again.
    bins.find_tail_pointers();

    // Insert the ascending suffixes.
    for i in (0..sa.len()).rev() {
        let sufi = sa[i];
        if sufi > 0 {
            let (lasti, lastc) = text.prev(sufi);
            if stypes.is_asc(lasti) {
                bins.tail_insert(sa, lasti, lastc);
            }
        }
    }
    debug!("after step 2: {:?}", sa);
    println!("step 2 complete");
}

struct SuffixTypes {
    types: Vec<SuffixType>,
}

#[derive(Clone, Copy, Eq, Show)]
enum SuffixType {
    Ascending,
    Descending,
    Valley,
}

impl SuffixTypes {
    fn new(num_bytes: u32) -> SuffixTypes {
        println!("allocating suffix types...");
        let mut stypes = Vec::with_capacity(num_bytes as usize);
        unsafe { stypes.set_len(num_bytes as usize); }
        println!("...done allocating suffix types.");
        SuffixTypes { types: stypes }
    }

    fn compute<'a, T>(&mut self, text: &T)
            where T: Text, <<T as Text>::IdxChars as Iterator>::Item: IdxChar {
        println!("finding suffix types");

        let mut chars = text.char_indices().map(|v| v.idx_char()).rev();

        // for t in self.types.iter_mut() { *t = Descending; }
        self.types[(text.len() - 1) as usize] = Descending;
        let (mut lasti, mut lastc) = chars.next().unwrap();
        for (i, c) in chars {
            if c < lastc {
                self.types[i] = Ascending;
            } else if c > lastc {
                self.types[i] = Descending;
            } else {
                self.types[i] = self.types[lasti].inherit();
            }
            if self.types[i].is_desc() && self.types[lasti].is_asc() {
                self.types[lasti] = Valley;
            }
            lastc = c;
            lasti = i;
        }
        for (i, t) in self.types.iter().enumerate() {
            debug!("suffix type for ({:?}, {:?}): {:?}",
                   i, text.char_at(i as u32), t);
        }
    }

    #[inline]
    fn ty(&self, i: u32) -> SuffixType { self.types[i as usize] }
    #[inline]
    fn is_asc(&self, i: u32) -> bool { self.ty(i).is_asc() }
    #[inline]
    fn is_desc(&self, i: u32) -> bool { self.ty(i).is_desc() }
    #[inline]
    fn is_valley(&self, i: u32) -> bool { self.ty(i).is_valley() }
    #[inline]
    fn equal(&self, i: u32, j: u32) -> bool { self.ty(i) == self.ty(j) }
}

impl SuffixType {
    #[inline]
    fn is_asc(&self) -> bool {
        match *self {
            Ascending | Valley => true,
            _ => false,
        }
    }

    #[inline]
    fn is_desc(&self) -> bool {
        if let Descending = *self { true } else { false }
    }

    #[inline]
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
    #[inline]
    fn eq(&self, other: &SuffixType) -> bool {
        (self.is_asc() && other.is_asc())
        || (self.is_desc() && other.is_desc())
    }
}

struct Bins {
    alphas: Vec<u32>,
    sizes: Vec<u32>,
    ptrs: BTreeMap<u32, u32>,
}

impl Bins {
    fn new() -> Bins {
        println!("Allocating bins...");
        let b = Bins {
            alphas: Vec::with_capacity(1_000),
            sizes: repeat(0).take(3_000_000).collect(),
            ptrs: BTreeMap::with_b(6),
        };
        println!("... done allocating bins.");
        b
    }

    fn find_sizes<I>(&mut self, mut chars: I) where I: Iterator<Item=u32> {
        self.ptrs.clear();

        println!("Find the size of each bin");
        unsafe { self.alphas.set_len(0); }
        for size in self.sizes.iter_mut() { *size = 0; }
        for c in chars {
            self.inc_size(c);
            if self.size(c) == 1 {
                self.alphas.push(c);
            }
        }
        self.alphas.sort();
        println!("alphabet size: {:?}", self.alphas.len());
    }

    fn find_head_pointers(&mut self) {
        let mut sum = 0u32;
        for &c in self.alphas.iter() {
            self.ptrs.insert(c, sum);
            sum += self.size(c);
        }
    }

    fn find_tail_pointers(&mut self) {
        let mut sum = 0u32;
        for &c in self.alphas.iter() {
            sum += self.size(c);
            self.ptrs.insert(c, sum - 1);
        }
    }

    #[inline]
    fn head_insert(&mut self, sa: &mut [u32], i: u32, c: u32) {
        let ptr = &mut self.ptrs[c];
        sa[*ptr as usize] = i;
        *ptr += 1;
    }

    #[inline]
    fn tail_insert(&mut self, sa: &mut [u32], i: u32, c: u32) {
        let ptr = &mut self.ptrs[c];
        sa[*ptr as usize] = i;
        *ptr -= 1;
    }

    #[inline] fn size(&self, c: u32) -> u32 { self.sizes[c as usize] }
    #[inline] fn inc_size(&mut self, c: u32) { self.sizes[c as usize] += 1; }
}

trait Text {
    type Chars: Iterator + DoubleEndedIterator;
    type IdxChars: Iterator + DoubleEndedIterator;
    fn len(&self) -> u32;
    fn prev(&self, i: u32) -> (u32, u32);
    fn next(&self, i: u32) -> (u32, u32);
    fn char_at(&self, i: u32) -> u32;
    fn chars(&self) -> Self::Chars;
    fn char_indices(&self) -> Self::IdxChars;
    fn wstring_equal(&self, i: u32, j: u32) -> bool;

    // For debugging. (This is why we aren't implementing slice syntax.)
    fn _string_from(&self, start: u32) -> String;
    fn _string(&self, start: u32, end: u32) -> String;
}

struct Unicode<'s> {
    s: &'s str,
    len: u32,
}

impl<'s> Unicode<'s> {
    fn from_str(s: &'s str) -> Unicode<'s> {
        Unicode::from_str_len(s, s.len() as u32)
    }

    fn from_str_len(s: &'s str, len: u32) -> Unicode<'s> {
        Unicode { s: s, len: len }
    }
}

impl<'s> Text for Unicode<'s> {
    type Chars = str::Chars<'s>;
    type IdxChars = str::CharIndices<'s>;

    #[inline]
    fn len(&self) -> u32 { self.len }

    #[inline]
    fn prev(&self, i: u32) -> (u32, u32) {
        let CharRange { ch, next } = self.s.char_range_at_reverse(i as usize);
        (next as u32, ch as u32)
    }

    #[inline]
    fn next(&self, i: u32) -> (u32, u32) {
        let CharRange { ch, next } = self.s.char_range_at(i as usize);
        (next as u32, ch as u32)
    }

    #[inline]
    fn char_at(&self, i: u32) -> u32 { self.s.char_at(i as usize) as u32 }

    fn chars(&self) -> str::Chars<'s> {
        self.s.chars()
    }

    fn char_indices(&self) -> str::CharIndices<'s> {
        self.s.char_indices()
    }

    fn wstring_equal(&self, i: u32, j: u32) -> bool { true }

    fn _string_from(&self, start: u32) -> String {
        self.s[start as usize..].to_owned()
    }

    fn _string(&self, start: u32, end: u32) -> String {
        self.s[start as usize .. end as usize].to_owned()
    }
}

struct LexNames<'s>(&'s [u32]);

impl<'s> Text for LexNames<'s> {
    type Chars = slice::Iter<'s, u32>;
    type IdxChars = iter::Enumerate<slice::Iter<'s, u32>>;

    #[inline]
    fn len(&self) -> u32 { self.0.len() as u32 }

    #[inline]
    fn prev(&self, i: u32) -> (u32, u32) { (i - 1, self.0[i as usize - 1]) }

    #[inline]
    fn next(&self, i: u32) -> (u32, u32) { (i + 1, self.0[i as usize + 1]) }

    #[inline]
    fn char_at(&self, i: u32) -> u32 { self.0[i as usize] }

    fn chars(&self) -> slice::Iter<'s, u32> {
        self.0.iter()
    }

    fn char_indices(&self) -> iter::Enumerate<slice::Iter<'s, u32>> {
        self.0.iter().enumerate()
    }

    fn wstring_equal(&self, i: u32, j: u32) -> bool { true }

    fn _string_from(&self, start: u32) -> String {
        format!("{:?}", &self.0[start as usize..])
    }

    fn _string(&self, start: u32, end: u32) -> String {
        format!("{:?}", &self.0[start as usize .. end as usize])
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

#[cfg(test)]
mod tests {
    extern crate test;

    use self::test::Bencher;
    use quickcheck::{TestResult, QuickCheck};
    use super::{naive, naive_table, sais_table, sais};

    #[test]
    fn sais_basic1() {
        assert_eq!(naive("apple"), sais("apple"));
    }

    #[test]
    fn sais_basic2() {
        assert_eq!(naive("banana"), sais("banana"));
    }

    #[test]
    fn sais_basic3() {
        assert_eq!(naive("mississippi"), sais("mississippi"));
    }

    #[test]
    fn qc_naive_equals_sais() {
        fn prop(s: String) -> TestResult {
            if s.is_empty() { return TestResult::discard(); }
            TestResult::from_bool(naive(&*s) == sais(&*s))
        }
        QuickCheck::new()
            .tests(1000)
            .max_tests(50000)
            .quickcheck(prop as fn(String) -> TestResult);
    }

    #[test]
    fn array_scratch() {
        // let s = "tgtgtgtgcaccg";
        // let s = "AGCTTTTCATTCT";
        let s = "QQR";
        let sa = sais_table(s);
        // let sa = sais("32P32Pz");

        debug!("\n\ngot suffix array:");
        for &i in sa.iter() {
            debug!("{:2}: {}", i, &s[i as usize..]);
        }
        debug!("\n\nnaive suffix array:");
        for &i in naive_table(&*s).iter() {
            debug!("{:2}: {}", i, &s[i as usize..]);
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
