use std::cmp::Ordering::{self, Equal, Greater, Less};
use std::collections::hash_map::{HashMap, Entry};
use std::iter::{range, repeat};

use SuffixArray;
use self::SuffixType::{Ascending, Descending, Valley};

const SENTINEL: u32 = 0x110000;
const INVALID: uint = 0x110001; // this is wrong

pub fn naive<'s>(s: &'s str) -> SuffixArray<'s> {
    make_suffix_array(s, naive_table(s))
}

pub fn naive_table<'s>(s: &'s str) -> Vec<uint> {
    let mut table: Vec<_> = s.char_indices().map(|(i, _)| i).collect();
    table.sort_by(|&a, &b| s[a..].cmp(s[b..]));
    table
}

fn make_suffix_array<'s>(s: &'s str, table: Vec<uint>) -> SuffixArray<'s> {
    let mut inverse: Vec<uint> = repeat(0).take(table.len()).collect();
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

fn lcp_lens_linear(text: &str, table: &[uint], inv: &[uint]) -> Vec<uint> {
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
        len += lcp_len(text[sufi1 + len..], text[sufi2 + len..]);
        lcps[rank] = len;
        if len > 0 {
            len -= 1;
        }
    }
    lcps
}

fn lcp_lens_quadratic(text: &str, table: &[uint]) -> Vec<uint> {
    // This is quadratic because there are N comparisons for each LCP.
    // But it is done in constant space.

    // The first LCP is always 0 because of the definition:
    //   LCP_LENS[i] = lcp_len(suf[i-1], suf[i])
    let mut lcps: Vec<_> = repeat(0).take(table.len()).collect();
    for (i, win) in table.windows(2).enumerate() {
        lcps[i+1] = lcp_len(text[win[0]..], text[win[1]..]);
    }
    lcps
}

/// Compute the length of the least common prefix between two strings.
fn lcp_len(a: &str, b: &str) -> uint {
    a.chars().zip(b.chars()).take_while(|&(ca, cb)| ca == cb).count()
}

pub fn sais<'s>(text: &'s str) -> SuffixArray<'s> {
    make_suffix_array(text, sais_table(text))
}

pub fn sais_table<'s>(text: &'s str) -> Vec<uint> {
    let mut chars = Vec::with_capacity(text.len());
    let mut chari = Vec::with_capacity(text.len());
    for (i, c) in text.char_indices() {
        chars.push(c as u32);
        chari.push(i);
    }

    // TODO: Figure out how to persist without the stupid sentinel.
    chars.push(SENTINEL);
    chari.push(chars.len());

    println!("calling sais_vec");
    let char_sa = sais_vec(&*chars);
    let mut byte_sa = Vec::with_capacity(char_sa.len());
    // Drop the first suffix, because it's always the sentinel, which we've
    // artificially inserted into the caller's string.
    for &char_sufstart in char_sa.iter().skip(1) {
        byte_sa.push(chari[char_sufstart]);
    }
    byte_sa
}

fn sais_vec(chars: &[u32]) -> Vec<uint> {
    println!("finding suffix types");
    let stypes = suffix_types(chars);

    // DEBUG.
    debug!("\nsuffix types");
    for (c, t) in chars.iter().zip(stypes.iter()) {
        debug!("{}: {}", chr(*c), t);
    }
    // DEBUG.

    println!("finding wstrings");
    let wstrs = find_wstrings(&*stypes, chars);

    // DEBUG.
    debug!("\nwstrings");
    for (i, wstr) in wstrs.iter().enumerate() {
        debug!("wstr {} [{}, {}): {}",
               i, wstr.start, wstr.end, chrs(wstr.slice(chars)));
    }
    // DEBUG.

    println!("sorting {} wstrings", wstrs.len());
    let mut wstrs_sorted = wstrs.clone();
    wstrs_sorted.sort_by(|w1, w2| wstring_cmp(&*chars, &*stypes, w1, w2));

    // DEBUG.
    debug!("\nsorted wstrings");
    for (i, wstr) in wstrs_sorted.iter().enumerate() {
        debug!("wstr {} [{}, {}): {}",
               i, wstr.start, wstr.end, chrs(wstr.slice(chars)));
    }
    // DEBUG.

    // Derive lexical names for each wstring. Each wstring gets its own
    // unique name.
    println!("building reduced string");
    let mut duplicate_names = false; // if stays false, we have our base case
    let mut cur_name = 0;
    let mut last_wstr = wstrs[0];
    let mut reduced: Vec<u32> = repeat(0).take(wstrs.len() + 1).collect();
    reduced[last_wstr.sequence] = cur_name;
    reduced[wstrs.len()] = SENTINEL;
    for &wstr in wstrs_sorted.iter().skip(1) {
        // let order = wstring_cmp(&*chars, &*stypes, &last_wstr, &wstr);
        // debug!("cmp({}, {}) == {}", last_wstr, wstr, order);
        if wstring_cmp(&*chars, &*stypes, &last_wstr, &wstr) == Equal {
            duplicate_names = true;
        } else {
            cur_name += 1;
            // never allow a name to be our sentinel
            if cur_name == SENTINEL {
                cur_name += 1;
            }
        }
        reduced[wstr.sequence] = cur_name;
        last_wstr = wstr;
    }

    // DEBUG.
    debug!("reduced string: {}", reduced);
    // DEBUG.

    if duplicate_names {
        println!("size of recursive case: {}, names: {}", reduced.len(), cur_name);
        let sa: Vec<uint> = sais_vec(&*reduced);
        // Drop the first suffix because it is always the sentinel.
        for (rank, &sufstart) in sa.iter().skip(1).enumerate() {
            wstrs_sorted[rank] = wstrs[sufstart];
        }

        // DEBUG.
        debug!("\nreduced suffixes");
        for (i, &sufstart) in sa.iter().enumerate() {
            debug!("{}: [{}..]: {}", i, sufstart, reduced[sufstart..]);
        }
        // DEBUG.
    }

    // DEBUG.
    debug!("\nsorted wstrings");
    for (i, wstr) in wstrs_sorted.iter().enumerate() {
        debug!("wstr {} [{}, {}): {}",
               i, wstr.start, wstr.end, chrs(wstr.slice(chars)));
    }
    // DEBUG.

    // TODO: This nonsense could probably be replaced with a BTreeMap.
    // But, we still have sentinels, which tweak the ordering of characters.
    // If we get rid of sentinels, a BTreeMap is trivial to use.
    // Otherwise, we need to newtype a character and define an ordering on it.
    let mut sa: Vec<int> = repeat(-1).take(chars.len()).collect();
    let mut bin_sizes: HashMap<u32, uint> = HashMap::new();
    for &c in chars.iter() {
        match bin_sizes.entry(c) {
            Entry::Vacant(v) => { v.set(1); }
            Entry::Occupied(mut v) => { *v.get_mut() += 1; }
        }
    }

    // Find the alphas in sorted order. These correspond to the labels of
    // each bin.
    let mut alphas: Vec<u32> = bin_sizes.keys().map(|&c| c).collect();
    alphas.sort_by(chrcmp);

    // These are pointers to the start/end of each bin. They are regenerated
    // at each step.
    let mut bin_ptrs: HashMap<u32, uint> = HashMap::new();

    // Find the index of the last element of each bin in `sa`.
    let mut sum = 0u;
    for &c in alphas.iter() {
        sum += bin_sizes[c];
        bin_ptrs.insert(c, sum - 1);
    }

    // Insert the valley suffixes.
    for wstr in wstrs_sorted.iter().rev() {
        let binp = &mut bin_ptrs[chars[wstr.start]];
        sa[*binp] = wstr.start as int;
        if *binp > 0 { *binp -= 1; }
    }
    debug!("after step 0: {}", sa);

    // Now find the start of each bin.
    let mut sum = 0u;
    for &c in alphas.iter() {
        bin_ptrs.insert(c, sum);
        sum += bin_sizes[c];
    }

    // Insert the descending suffixes.
    for i in range(0, sa.len()) {
        let sufi = sa[i];
        if sufi > 0 && stypes[(sufi - 1) as uint].is_desc() {
            let binp = &mut bin_ptrs[chars[(sufi - 1) as uint]];
            sa[*binp] = sufi - 1;
            *binp += 1;
        }
    }
    debug!("after step 1: {}", sa);

    // ... and find the end of each bin again.
    let mut sum = 0u;
    for &c in alphas.iter() {
        sum += bin_sizes[c];
        bin_ptrs.insert(c, sum - 1);
    }

    // Insert the ascending suffixes.
    for i in range(0, sa.len()).rev() {
        let sufi = sa[i];
        if sufi > 0 && stypes[(sufi - 1) as uint].is_asc() {
            let binp = &mut bin_ptrs[chars[(sufi - 1) as uint]];
            sa[*binp] = sufi - 1;
            *binp -= 1;
        }
    }
    debug!("after step 2: {}", sa);
    unsafe { ::std::mem::transmute(sa) }
}

#[derive(Clone, Copy, Eq, Ord, Show)]
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
        self.is_asc() && other.is_asc()
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

fn suffix_types(chars: &[u32]) -> Vec<SuffixType> {
    let mut stypes: Vec<_> = repeat(Descending).take(chars.len()).collect();
    stypes[chars.len() - 1] = Valley;
    for i in range(0, chars.len() - 2).rev() {
        if chars[i] < chars[i+1] {
            stypes[i] = Ascending;
        } else if chars[i] > chars[i+1] {
            stypes[i] = Descending;
        } else {
            stypes[i] = stypes[i+1].inherit();
        }
        if stypes[i].is_desc() && stypes[i+1].is_asc() {
            stypes[i+1] = Valley;
        }
    }
    stypes
}

#[derive(Copy, Clone, Show)]
struct WString {
    sequence: uint,
    start: uint,
    end: uint,
}

impl WString {
    fn start_at(sequence: uint, i: uint) -> WString {
        WString { sequence: sequence, start: i, end: 0 }
    }

    fn index<'s, T>(&self, xs: &'s [T], i: uint) -> &'s T {
        assert!(self.start + i < self.end);
        &xs[self.start + i]
    }

    fn slice<'s, T>(&self, xs: &'s [T]) -> &'s [T] {
        xs[self.start .. self.end]
    }

    fn len(&self) -> uint {
        self.end - self.start
    }
}

fn find_wstrings(stypes: &[SuffixType], chars: &[u32]) -> Vec<WString> {
    let mut wstrs: Vec<WString> = vec![];
    let mut sequence = 0;
    let mut cur = WString::start_at(sequence, 0);
    for (i, (c, t)) in chars.iter().zip(stypes.iter()).enumerate() {
        if t.is_valley() {
            if cur.start == 0 {
                cur.start = i;
            } else {
                cur.end = i + 1;
                wstrs.push(cur);
                sequence += 1;
                cur = WString::start_at(sequence, i);
            }
        }
    }
    // We might need to do something different here if we don't have
    // a sentinel.
    // Although... Since we've converted everything to `u32`, we could just
    // pick an arbitrary `u32` that is not a valid Unicode scalar value. ---AG
    cur.end = chars.len();
    wstrs.push(cur);
    wstrs
}

fn wstring_cmp(
    chars: &[u32],
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
        match chrcmp(w1.index(chars, i), w2.index(chars, i)) {
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
    if n == SENTINEL {
        '$'
    } else if n == INVALID as u32 { // wrong
        '?'
    } else {
        ::std::char::from_u32(n).unwrap()
    }
}

#[inline]
fn chrcmp(c1: &u32, c2: &u32) -> Ordering {
    if *c1 == SENTINEL && *c2 != SENTINEL {
        Less
    } else if *c1 != SENTINEL && *c2 == SENTINEL {
        Greater
    } else {
        c1.cmp(c2)
    }
}

#[cfg(test)]
mod tests {
    extern crate test;

    use self::test::Bencher;
    use quickcheck::{TestResult, quickcheck};
    use super::{naive, naive_table, sais, sais_table};

    #[test]
    fn basic() {
        let sa = naive("banana");
        debug!("{}", sa);
        assert_eq!(sa.suffix(0), "a");
        assert_eq!(sa.suffix(1), "ana");
        assert_eq!(sa.suffix(2), "anana");
        assert_eq!(sa.suffix(3), "banana");
        assert_eq!(sa.suffix(4), "na");
        assert_eq!(sa.suffix(5), "nana");
        assert_eq!(sa.lcp(0), "");
        assert_eq!(sa.lcp(1), "a");
        assert_eq!(sa.lcp(2), "ana");
        assert_eq!(sa.lcp(3), "");
        assert_eq!(sa.lcp(4), "");
        assert_eq!(sa.lcp(5), "na");
    }

    #[test]
    fn basic2() {
        let sa = naive("apple");
        debug!("{}", sa);
        assert_eq!(sa.suffix(0), "apple");
        assert_eq!(sa.suffix(1), "e");
        assert_eq!(sa.suffix(2), "le");
        assert_eq!(sa.suffix(3), "ple");
        assert_eq!(sa.suffix(4), "pple");
        assert_eq!(sa.lcp(0), "");
        assert_eq!(sa.lcp(1), "");
        assert_eq!(sa.lcp(2), "");
        assert_eq!(sa.lcp(3), "");
        assert_eq!(sa.lcp(4), "p");
    }

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
        quickcheck(prop as fn(String) -> TestResult);
    }

    #[test]
    fn array_scratch() {
        let sa = sais("tgtgtgtgcaccg");
        debug!("{}", sa);

        assert_eq!(sa, naive("tgtgtgtgcaccg"));
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
