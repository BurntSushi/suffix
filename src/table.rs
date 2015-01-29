
use std::borrow::ToOwned;
use std::cmp::Ordering::{self, Equal, Greater, Less};
use std::collections::btree_map::{BTreeMap, Entry};
use std::iter::{self, repeat};
use std::mem::transmute;
use std::num::Int;
use std::slice;
use std::str::{self, CharRange};
use std::u32;

use {SuffixArray, vec_from_elem};

use self::SuffixType::{Ascending, Descending, Valley};

#[derive(Clone, Eq, PartialEq)]
pub struct SuffixTable<'s> {
    text: &'s str,
    table: Vec<u32>,
}

impl<'s> SuffixTable<'s> {
    pub fn new(text: &'s str) -> SuffixTable<'s> {
        SuffixTable {
            text: text,
            table: sais_table(text),
        }
    }

    pub fn new_naive(text: &'s str) -> SuffixTable<'s> {
        SuffixTable {
            text: text,
            table: naive_table(text),
        }
    }

    pub fn into_suffix_array(self) -> SuffixArray<'s> {
        SuffixArray::from_table(self)
    }

    pub fn table(&self) -> &[u32] { self.table.as_slice() }

    pub fn text(&self) -> &'s str { self.text }

    #[inline]
    pub fn len(&self) -> usize { self.table.len() }

    #[inline]
    pub fn suffix(&self, i: usize) -> &str {
        &self.text[self.table[i] as usize..]
    }
}

fn naive_table(text: &str) -> Vec<u32> {
    let mut table = Vec::with_capacity(text.len() / 2);
    for (ci, _) in text.char_indices() { table.push(ci as u32); }
    if table.len() > 0 {
        assert!(table[table.len() - 1] <= u32::MAX);
    }
    table.sort_by(|&a, &b| text[a as usize..].cmp(&text[b as usize..]));
    table
}

pub fn sais_table<'s>(text: &'s str) -> Vec<u32> {
    let chars = text.chars().count();
    assert!(chars as u32 <= u32::MAX);
    let mut sa = vec_from_elem(chars, 0u32);

    let mut stypes = SuffixTypes::new(text.len() as u32);
    let mut bins = Bins::new();

    sais(&mut *sa, &mut stypes, &mut bins, &Unicode::from_str(text));
    sa
}

fn sais<T>(sa: &mut [u32], stypes: &mut SuffixTypes, bins: &mut Bins, text: &T)
        where T: Text, <<T as Text>::IdxChars as Iterator>::Item: IdxChar {
    // Instead of working out edge cases in the code below, just allow them
    // to assume >=2 characters.
    match text.len() {
        0 => return,
        1 => { sa[0] = 0; return; }
        _ => {},
    }

    for v in sa.iter_mut() { *v = 0; }
    stypes.compute(text);
    bins.find_sizes(text.char_indices().map(|c| c.idx_char().1));
    bins.find_tail_pointers();

    // Insert the valley suffixes.
    for (i, c) in text.char_indices().map(|v| v.idx_char()) {
        if stypes.is_valley(i as u32) {
            bins.tail_insert(sa, i as u32, c);
        }
    }

    // Now find the start of each bin.
    bins.find_head_pointers();

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

    // ... and the find the end of each bin.
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

    // Find and move all wstrings to the beinning of `sa`.
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
        let cur_sufi = sa[i as usize];
        if prev_sufi == 0 || !text.wstring_equal(stypes, cur_sufi, prev_sufi) {
            name += 1;
            prev_sufi = cur_sufi;
        }
        // This divide-by-2 trick only works because it's impossible to have
        // two wstrings start at adjacent locations (they must at least be
        // separated by a single descending character).
        sa[(num_wstrs + (cur_sufi / 2)) as usize] = name - 1;
    }

    // We've inserted the lexical names into the latter half of the suffix
    // array, but it's sparse. so let's smush them all up to the end.
    let mut j = sa.len() as u32 - 1;
    for i in (num_wstrs..(sa.len() as u32)).rev() {
        if sa[i as usize] != u32::MAX {
            sa[j as usize] = sa[i as usize];
            j -= 1;
        }
    }

    // If we have fewer names than wstrings, then there are at least 2
    // equivalent wstrings, which means we need to recurse and sort them.
    if name < num_wstrs {
        let split_at = sa.len() - (num_wstrs as usize);
        let (r_sa, r_text) = sa.split_at_mut(split_at);
        sais(&mut r_sa[..num_wstrs as usize], stypes, bins, &LexNames(r_text));
        stypes.compute(text);
    } else {
        for i in (0..num_wstrs) {
            let reducedi = sa[((sa.len() as u32) - num_wstrs + i) as usize];
            sa[reducedi as usize] = i;
        }
    }

    // Re-calibrate the bins by finding their sizes and the end of each bin.
    bins.find_sizes(text.char_indices().map(|c| c.idx_char().1));
    bins.find_tail_pointers();

    // Replace the lexical names with their corresponding suffix index in the
    // original text.
    let mut j = sa.len() - (num_wstrs as usize);
    for (i, c) in text.char_indices().map(|v| v.idx_char()) {
        if stypes.is_valley(i as u32) {
            sa[j] = i as u32;
            j += 1;
        }
    }
    // And now map the suffix indices from the reduced text to suffix
    // indices in the original text. Remember, `sa[i]` yields a lexical name.
    // So all we have to do is get the suffix index of the original text for
    // that lexical name (which was made possible in the loop above).
    //
    // In other words, this sets the suffix indices of only the wstrings.
    for i in (0..num_wstrs) {
        let sufi = sa[i as usize];
        sa[i as usize] = sa[(sa.len() as u32 - num_wstrs + sufi) as usize];
    }
    // Now zero out everything after the wstrs.
    for i in (num_wstrs..(sa.len() as u32)) {
        sa[i as usize] = 0;
    }

    // Insert the valley suffixes and zero out everything else..
    for i in (0..num_wstrs).rev() {
        let sufi = sa[i as usize];
        sa[i as usize] = 0;
        bins.tail_insert(sa, sufi, text.char_at(sufi));
    }

    // Now find the start of each bin.
    bins.find_head_pointers();

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
        let mut stypes = Vec::with_capacity(num_bytes as usize);
        unsafe { stypes.set_len(num_bytes as usize); }
        SuffixTypes { types: stypes }
    }

    fn compute<'a, T>(&mut self, text: &T)
            where T: Text, <<T as Text>::IdxChars as Iterator>::Item: IdxChar {
        let mut chars = text.char_indices().map(|v| v.idx_char()).rev();
        let (mut lasti, mut lastc) = match chars.next() {
            None => return,
            Some(t) => t,
        };
        self.types[lasti] = Descending;
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
    ptrs: Vec<u32>,
}

impl Bins {
    fn new() -> Bins {
        Bins {
            alphas: Vec::with_capacity(10_000),
            sizes: Vec::with_capacity(10_000),
            ptrs: Vec::new(), // re-allocated later, no worries
        }
    }

    fn find_sizes<I>(&mut self, mut chars: I) where I: Iterator<Item=u32> {
        unsafe { self.alphas.set_len(0); }
        for size in self.sizes.iter_mut() { *size = 0; }
        for c in chars {
            self.inc_size(c);
            if self.size(c) == 1 {
                self.alphas.push(c);
            }
        }
        self.alphas.sort();

        let ptrs_len = self.alphas[self.alphas.len() - 1] + 1;
        self.ptrs = vec_from_elem(ptrs_len as usize, 0u32);
    }

    fn find_head_pointers(&mut self) {
        let mut sum = 0u32;
        for &c in self.alphas.iter() {
            self.ptrs[c as usize] = sum;
            sum += self.size(c);
        }
    }

    fn find_tail_pointers(&mut self) {
        let mut sum = 0u32;
        for &c in self.alphas.iter() {
            sum += self.size(c);
            self.ptrs[c as usize] = sum - 1;
        }
    }

    #[inline]
    fn head_insert(&mut self, sa: &mut [u32], i: u32, c: u32) {
        let ptr = &mut self.ptrs[c as usize];
        sa[*ptr as usize] = i;
        *ptr += 1;
    }

    #[inline]
    fn tail_insert(&mut self, sa: &mut [u32], i: u32, c: u32) {
        let ptr = &mut self.ptrs[c as usize];
        sa[*ptr as usize] = i;
        *ptr -= 1;
    }

    #[inline]
    fn inc_size(&mut self, c: u32) {
        if c as usize >= self.sizes.len() {
            let (len, new_len) = (self.sizes.len(), 1 + (c as usize));
            self.sizes.reserve(new_len - len);
            unsafe { self.sizes.set_len(new_len); }
            for v in self.sizes[len..new_len].iter_mut() { *v = 0; }
        }
        self.sizes[c as usize] += 1;
    }

    #[inline]
    fn size(&self, c: u32) -> u32 { self.sizes[c as usize] }
}

trait Text {
    type IdxChars: Iterator + DoubleEndedIterator;
    fn len(&self) -> u32;
    fn prev(&self, i: u32) -> (u32, u32);
    fn char_at(&self, i: u32) -> u32;
    fn char_indices(&self) -> Self::IdxChars;
    fn wstring_equal(&self, stypes: &SuffixTypes, w1: u32, w2: u32) -> bool;
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
    type IdxChars = str::CharIndices<'s>;

    #[inline]
    fn len(&self) -> u32 { self.len }

    #[inline]
    fn prev(&self, i: u32) -> (u32, u32) {
        let CharRange { ch, next } = self.s.char_range_at_reverse(i as usize);
        (next as u32, ch as u32)
    }

    #[inline]
    fn char_at(&self, i: u32) -> u32 { self.s.char_at(i as usize) as u32 }

    fn char_indices(&self) -> str::CharIndices<'s> {
        self.s.char_indices()
    }

    fn wstring_equal(&self, stypes: &SuffixTypes, w1: u32, w2: u32) -> bool {
        let w1chars = self.s[w1 as usize..].char_indices();
        let w2chars = self.s[w2 as usize..].char_indices();
        for ((i1, c1), (i2, c2)) in w1chars.zip(w2chars) {
            let (i1, i2) = (w1 + i1 as u32, w2 + i2 as u32);
            if c1 != c2 || !stypes.equal(i1, i2) {
                return false;
            }
            if i1 > w1 && (stypes.is_valley(i1) || stypes.is_valley(i2)) {
                return true;
            }
        }
        // At this point, we've exhausted either `w1` or `w2`, which means the
        // next character for one of them should be the sentinel. Since
        // `w1 != w2`, only one string can be exhausted. The sentinel is never
        // equal to another character, so we can conclude that the wstrings
        // are not equal.
        false
    }
}

struct LexNames<'s>(&'s [u32]);

impl<'s> Text for LexNames<'s> {
    type IdxChars = iter::Enumerate<slice::Iter<'s, u32>>;

    #[inline]
    fn len(&self) -> u32 { self.0.len() as u32 }

    #[inline]
    fn prev(&self, i: u32) -> (u32, u32) { (i - 1, self.0[i as usize - 1]) }

    #[inline]
    fn char_at(&self, i: u32) -> u32 { self.0[i as usize] }

    fn char_indices(&self) -> iter::Enumerate<slice::Iter<'s, u32>> {
        self.0.iter().enumerate()
    }

    fn wstring_equal(&self, stypes: &SuffixTypes, w1: u32, w2: u32) -> bool {
        let w1chars = self.0[w1 as usize..].iter().enumerate();
        let w2chars = self.0[w2 as usize..].iter().enumerate();
        for ((i1, c1), (i2, c2)) in w1chars.zip(w2chars) {
            let (i1, i2) = (w1 + i1 as u32, w2 + i2 as u32);
            if c1 != c2 || !stypes.equal(i1, i2) {
                return false;
            }
            if i1 > w1 && (stypes.is_valley(i1) || stypes.is_valley(i2)) {
                return true;
            }
        }
        // At this point, we've exhausted either `w1` or `w2`, which means the
        // next character for one of them should be the sentinel. Since
        // `w1 != w2`, only one string can be exhausted. The sentinel is never
        // equal to another character, so we can conclude that the wstrings
        // are not equal.
        false
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

#[cfg(test)]
mod tests {
    extern crate test;

    use self::test::Bencher;
    use quickcheck::{TestResult, QuickCheck};
    use super::{SuffixTable, naive_table, sais_table};

    #[test]
    fn sais_basic1() {
        assert_eq!(naive_table("apple"), sais_table("apple"));
    }

    #[test]
    fn sais_basic2() {
        assert_eq!(naive_table("banana"), sais_table("banana"));
    }

    #[test]
    fn sais_basic3() {
        assert_eq!(naive_table("mississippi"), sais_table("mississippi"));
    }

    #[test]
    fn sais_basic4() {
        assert_eq!(naive_table("tgtgtgtgcaccg"), sais_table("tgtgtgtgcaccg"));
    }

    #[test]
    fn qc_naive_equals_sais() {
        fn prop(s: String) -> TestResult {
            if s.is_empty() { return TestResult::discard(); }
            let expected = SuffixTable::new_naive(&*s);
            let got = SuffixTable::new(&*s);
            TestResult::from_bool(expected == got)
        }
        QuickCheck::new()
            .tests(1000)
            .max_tests(50000)
            .quickcheck(prop as fn(String) -> TestResult);
    }

    #[test]
    fn array_scratch() {
        let s = "tgtgtgtgcaccg";
        let sa = sais_table(s);

        debug!("\n\ngot suffix array:");
        for &i in sa.iter() {
            debug!("{:2}: {}", i, &s[i as usize..]);
        }
        debug!("\n\nnaive suffix array:");
        for &i in naive_table(&*s).iter() {
            debug!("{:2}: {}", i, &s[i as usize..]);
        }
    }

    #[bench]
    fn naive_small(b: &mut Bencher) {
        let s = "mississippi";
        b.iter(|| { naive_table(s); })
    }

    #[bench]
    fn sais_small(b: &mut Bencher) {
        let s = "mississippi";
        b.iter(|| { sais_table(s); })
    }

    #[bench]
    fn naive_dna_small(b: &mut Bencher) {
        let s = include_str!("AP009048_10000.fasta");
        b.iter(|| { naive_table(s); })
    }

    #[bench]
    fn sais_dna_small(b: &mut Bencher) {
        let s = include_str!("AP009048_10000.fasta");
        b.iter(|| { sais_table(s); })
    }

    #[bench]
    fn naive_dna_medium(b: &mut Bencher) {
        let s = include_str!("AP009048_100000.fasta");
        b.iter(|| { naive_table(s); })
    }

    #[bench]
    fn sais_dna_medium(b: &mut Bencher) {
        let s = include_str!("AP009048_100000.fasta");
        b.iter(|| { sais_table(s); })
    }
}
