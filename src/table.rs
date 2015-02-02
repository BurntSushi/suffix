use std::borrow::IntoCow;
use std::cmp;
use std::fmt;
use std::iter;
use std::slice;
use std::str;
use std::string::CowString;
use std::u32;

use {SuffixArray, binary_search, vec_from_elem};

use self::SuffixType::{Ascending, Descending, Valley};

#[derive(Clone, Eq, PartialEq)]
pub struct SuffixTable<'s> {
    text: CowString<'s>,
    table: Vec<u32>,
}

impl<'s> SuffixTable<'s> {
    pub fn new<S>(text: S) -> SuffixTable<'s>
            where S: IntoCow<'s, String, str> {
        let text = text.into_cow();
        let table = sais_table(&*text);
        SuffixTable {
            text: text,
            table: table,
        }
    }

    #[doc(hidden)]
    pub fn new_naive<S>(text: S) -> SuffixTable<'s>
            where S: IntoCow<'s, String, str> {
        let text = text.into_cow();
        let table = naive_table(&*text);
        SuffixTable {
            text: text,
            table: table,
        }
    }

    #[doc(hidden)]
    pub fn into_suffix_array(self) -> SuffixArray<'s> {
        SuffixArray::from_table(self)
    }

    #[inline]
    pub fn table(&self) -> &[u32] { self.table.as_slice() }

    #[inline]
    pub fn text(&self) -> &str { &*self.text }

    #[inline]
    pub fn len(&self) -> usize { self.table.len() }

    #[inline]
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    #[inline]
    pub fn suffix(&self, i: usize) -> &str {
        &self.text[self.table[i] as usize..]
    }

    pub fn contains(&self, query: &str) -> bool {
        query.len() > 0 && self.table.binary_search_by(|&sufi| {
            let sufi = sufi as usize;
            let len = cmp::min(query.len(), self.text.len() - sufi);
            self.text[sufi..(sufi + len)].cmp(query)
        }).is_ok()
    }

    pub fn positions(&self, query: &str) -> &[u32] {
        // We can quickly decide whether the query won't match at all if
        // it's outside the range of suffixes.
        if self.len() == 0
           || query.len() == 0
           || (query < self.suffix(0) && !self.suffix(0).starts_with(query))
           || query > self.suffix(self.len() - 1) {
            return &[];
        }

        // The below is pretty close to the algorithm on Wikipedia:
        //
        //     http://en.wikipedia.org/wiki/Suffix_array#Applications
        //
        // The key difference is that after we find the start index, we look
        // for the end by finding the first occurrence that doesn't start
        // with `query`. That becomes our upper bound.
        let start = binary_search(&*self.table,
            |&sufi| query <= &self.text[sufi as usize..]);
        // Hmm, we could inline this second binary search and start its
        // "left" point with `start` from above. Probably not a huge difference
        // in practice though. ---AG
        let end = binary_search(&*self.table,
            |&sufi| !self.text[sufi as usize..].starts_with(query));
        // lg!("query: {:?}, start: {:?}, end: {:?}", query, start, end);

        // Whoops. If start is somehow greater than end, then we've got
        // nothing.
        if start > end {
            return &[];
        }
        &self.table[start..end]
    }
}

impl<'s> fmt::Debug for SuffixTable<'s> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(writeln!(f, "\n-----------------------------------------"));
        try!(writeln!(f, "SUFFIX TABLE"));
        try!(writeln!(f, "text: {}", self.text()));
        for (rank, &sufstart) in self.table.iter().enumerate() {
            try!(writeln!(f, "suffix[{}] {}, {}",
                          rank, sufstart, self.suffix(rank)));
        }
        writeln!(f, "-----------------------------------------")
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

fn sais_table<'s>(text: &'s str) -> Vec<u32> {
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
    for (i, _) in text.char_indices().map(|v| v.idx_char()) {
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

#[derive(Copy, Debug, Eq)]
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
        let str::CharRange { ch, next } =
            self.s.char_range_at_reverse(i as usize);
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

// #[cfg(test)]
// mod tests {
    // use super::SuffixTable;
//
    // #[test]
    // fn array_scratch() {
        // let s = "tgtgtgtgcaccg";
        // let table = SuffixTable::new(s);
        // lg!("{:?}", table);
        // lg!("SEARCH: {:?}", table.positions("gtg"));
    // }
// }
