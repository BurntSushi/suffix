use std::borrow::{Cow, IntoCow};
use std::fmt;
use std::iter::{self, order, AdditiveIterator};
use std::slice;
use std::str;
use std::u32;

use {SuffixTree, binary_search, vec_from_elem};
use array::SuffixArray;
use tree::to_suffix_tree;

use self::SuffixType::{Ascending, Descending, Valley};

/// A suffix table is a sequence of lexicographically sorted suffixes.
///
/// The lifetime `'s` refers to the text when borrowed.
///
/// This is distinct from a suffix array in that it *only* contains
/// suffix indices. It has no "enhanced" information like the inverse suffix
/// table or least-common-prefix lengths (LCP array). This representation
/// limits what you can do (and how fast), but it uses very little memory
/// (4 bytes per character in the text).
///
/// # Construction
///
/// Suffix array construction is done in `O(n)` time and in `O(kn)` space,
/// where `k` is the number of unique characters in the text. (More details
/// below.) The specific algorithm implemented is from
/// [(Nong et al., 2009)](https://local.ugene.unipro.ru/tracker/secure/attachment/12144/Linear%20Suffix%20Array%20Construction%20by%20Almost%20Pure%20Induced-Sorting.pdf),
/// but I actually used the description found in
/// [(Shrestha et al., 2014)](http://bib.oxfordjournals.org/content/15/2/138.full.pdf),
/// because it is much more accessible to someone who is not used to reading
/// algorithms papers.
///
/// The main thrust of the algorithm is that of "reduce and conquer." Namely,
/// it reduces the problem of finding lexicographically sorted suffixes to a
/// smaller subproblem, and solves it recursively. The subproblem is to find
/// the suffix array of a smaller string, where that string is composed by
/// naming contiguous regions of the original text. If there are any duplicate
/// names, then the algorithm procedes recursively. If there are no duplicate
/// names (base case), then the suffix array of the subproblem is already
/// computed. In essence, this "inductively sorts" suffixes of the original
/// text with several linear scans over the text. Because of the number of
/// linear scans, the performance of construction is heavily tied to cache
/// performance (and this is why `u32` is used to represent the suffix index
/// instead of a `u64`).
///
/// The space usage is roughly `6` bytes per character. (The optimal bound is
/// `5` bytes per character, although that may be for a small constant
/// alphabet.) `4` bytes comes from the suffix array itself. The extra `2`
/// bytes comes from storing the suffix type of each character (`1` byte) and
/// information about bin boundaries, where the number of bins is equal to
/// the number of unique characters in the text. This doesn't formally imply
/// another byte of overhead, but in practice, the alphabet can get quite large
/// when solving the subproblems mentioned above (even if the alphabet of the
/// original text is very small).
#[derive(Clone, Eq, PartialEq)]
pub struct SuffixTable<'s> {
    text: Cow<'s, str>,
    table: Vec<u32>,
}

impl<'s> SuffixTable<'s> {
    /// Creates a new suffix table for `text` in `O(n)` time and `O(kn)`
    /// space, where `k` is the size of the alphabet in the text.
    ///
    /// The table stores either `S` or a `&S` and a lexicographically sorted
    /// list of suffixes. Each suffix is represented by a 32 bit integer and
    /// is a **byte index** into `text`.
    ///
    /// # Panics
    ///
    /// Panics if the `text` contains more than `2^32 - 1` bytes. This
    /// restriction is mostly artificial; there's no fundamental reason why
    /// suffix array construction algorithm can't use a `u64`. Nevertheless,
    /// `u32` was chosen for performance reasons. The performance of the
    /// construction algorithm is highly dependent on cache performance, which
    /// is degraded with a bigger number type. `u32` strikes a nice balance; it
    /// gets good performance while allowing most reasonably sized documents
    /// (~4GB).
    pub fn new<S>(text: S) -> SuffixTable<'s> where S: IntoCow<'s, str> {
        let text = text.into_cow();
        let table = sais_table(&text);
        SuffixTable {
            text: text,
            table: table,
        }
    }

    /// The same as `new`, except it runs in `O(n^2 * logn)` time.
    ///
    /// This is a simple naive implementation that sorts the suffixes. This
    /// tends to have lower overhead, so it can be useful when creating lots
    /// of suffix tables for small strings.
    #[doc(hidden)]
    pub fn new_naive<S>(text: S) -> SuffixTable<'s> where S: IntoCow<'s, str> {
        let text = text.into_cow();
        let table = naive_table(&text);
        SuffixTable {
            text: text,
            table: table,
        }
    }

    /// Creates a new suffix table from an existing list of lexicographically
    /// sorted suffix indices.
    ///
    /// Note that the invariant that `table` must be a suffix table of `text`
    /// is not checked! If it isn't, this will cause other operations on a
    /// suffix table to fail in weird ways.
    ///
    /// Note that if `table` is borrowed (i.e., a `&[u8]`), then it is copied.
    ///
    /// This fails if the number of characters in `text` does not equal the
    /// number of suffixes in `table`.
    pub fn from_parts<'t, S, T>(text: S, table: T) -> SuffixTable<'s>
            where S: IntoCow<'s, str>, T: IntoCow<'t, [u32]> {
        let (text, table) = (text.into_cow(), table.into_cow());
        assert_eq!(text.chars().count(), table.len());
        SuffixTable {
            text: text,
            table: table.into_owned(),
        }
    }

    /// Extract the parts of a suffix table.
    ///
    /// This is useful to avoid copying when the suffix table is part of an
    /// intermediate computation.
    pub fn into_parts(self) -> (Cow<'s, str>, Vec<u32>) {
        (self.text, self.table)
    }

    /// Converts this suffix table to an enhanced suffix array.
    ///
    /// Not ready yet.
    #[doc(hidden)]
    pub fn into_suffix_array(self) -> SuffixArray<'s> {
        SuffixArray::from_table(self)
    }

    /// Creates a suffix tree in linear time and space.
    pub fn to_suffix_tree(&'s self) -> SuffixTree<'s> {
        to_suffix_tree(self)
    }

    /// Computes the LCP array in linear time and linear space.
    pub fn lcp_lens(&self) -> Vec<u32> {
        let mut inverse = vec_from_elem(self.text.len(), 0u32);
        for (rank, &sufstart) in self.table().iter().enumerate() {
            inverse[sufstart as usize] = rank as u32;
        }
        lcp_lens_quadratic(self.text(), self.table())
        // Broken on Unicode text for now. ---AG
        // lcp_lens_linear(self.text(), self.table(), &inverse)
    }

    /// Return the suffix table.
    #[inline]
    pub fn table(&self) -> &[u32] { &self.table }

    /// Return the text.
    #[inline]
    pub fn text(&self) -> &str { &self.text }

    /// Returns the number of suffixes in the table.
    ///
    /// Alternatively, this is the number of characters in the text.
    #[inline]
    pub fn len(&self) -> usize { self.table.len() }

    /// Returns `true` iff `self.len() == 0`.
    #[inline]
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// Returns the suffix at index `i`.
    #[inline]
    pub fn suffix(&self, i: usize) -> &str {
        &self.text[self.table[i] as usize..]
    }

    /// Returns true if and only if `query` is in text.
    ///
    /// This runs in `O(mlogn)` time, where `m == query.len()` and
    /// `n == self.len()`. (As far as this author knows, this is the best known
    /// bound for a plain suffix table.)
    ///
    /// You should prefer this over `positions` when you only need to test
    /// existence (because it is faster).
    ///
    /// # Example
    ///
    /// Build a suffix array of some text and test existence of a substring:
    ///
    /// ```rust
    /// use suffix::SuffixTable;
    ///
    /// let sa = SuffixTable::new("The quick brown fox.");
    /// assert!(sa.contains("quick"));
    /// ```
    pub fn contains(&self, query: &str) -> bool {
        let nquery = query.chars().count();
        nquery > 0 && self.table.binary_search_by(|&sufi| {
            order::cmp(self.text[sufi as usize..].chars().take(nquery),
                       query.chars())
        }).is_ok()
    }

    /// Returns an unordered list of positions where `query` starts in `text`.
    ///
    /// This runs in `O(mlogn)` time, where `m == query.len()` and
    /// `n == self.len()`. (As far as this author knows, this is the best known
    /// bound for a plain suffix table.)
    ///
    /// Positions are byte indices into `text`.
    ///
    /// If you just need to test existence, then use `contains` since it is
    /// faster.
    ///
    /// # Example
    ///
    /// Build a suffix array of some text and find all occurrences of a
    /// substring:
    ///
    /// ```rust
    /// use suffix::SuffixTable;
    ///
    /// let sa = SuffixTable::new("The quick brown fox was very quick.");
    /// assert_eq!(sa.positions("quick"), vec![4, 29]);
    /// ```
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
        let start = binary_search(&self.table,
            |&sufi| query <= &self.text[sufi as usize..]);
        let end = start + binary_search(&self.table[start..],
            |&sufi| !self.text[sufi as usize..].starts_with(query));

        // Whoops. If start is somehow greater than end, then we've got
        // nothing.
        if start > end {
            &[]
        } else {
            &self.table[start..end]
        }
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

#[allow(dead_code)]
fn lcp_lens_linear(text: &str, table: &[u32], inv: &[u32]) -> Vec<u32> {
    // This algorithm is bunk because it doesn't work on Unicode. See comment
    // in the code below.

    // This is a linear time construction algorithm taken from the first
    // two slides of:
    // http://www.cs.helsinki.fi/u/tpkarkka/opetus/11s/spa/lecture10.pdf
    //
    // It does require the use of the inverse suffix array, which makes this
    // O(n) in space. The inverse suffix array gives us a special ordering
    // with which to compute the LCPs.
    let mut lcps = vec_from_elem(table.len(), 0u32);
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
            // This is an illegal move because `len` is derived from `text`,
            // which is a Unicode string. Subtracting `1` here assumes every
            // character is a single byte in UTF-8, which is obviously wrong.
            // TODO: Figure out how to get LCP lengths in linear time on
            // UTF-8 encoded strings.
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
    let mut lcps = vec_from_elem(table.len(), 0u32);
    for (i, win) in table.windows(2).enumerate() {
        lcps[i+1] =
            lcp_len(&text[win[0] as usize..], &text[win[1] as usize..]);
    }
    lcps
}

fn lcp_len(a: &str, b: &str) -> u32 {
    use std::iter::AdditiveIterator;
    a.chars()
     .zip(b.chars())
     .take_while(|&(ca, cb)| ca == cb)
     .map(|(c, _)| c.len_utf8())
     .sum() as u32
}

fn naive_table(text: &str) -> Vec<u32> {
    let mut table = Vec::with_capacity(text.len() / 2);
    let mut count = 0usize;
    for (ci, _) in text.char_indices() { table.push(ci as u32); count += 1; }
    assert!(count <= u32::MAX as usize);

    table.sort_by(|&a, &b| text[a as usize..].cmp(&text[b as usize..]));
    table
}

fn sais_table<'s>(text: &'s str) -> Vec<u32> {
    let chars = text.chars().count();
    assert!(chars <= u32::MAX as usize);
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

    fn find_sizes<I>(&mut self, chars: I) where I: Iterator<Item=u32> {
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
        if *ptr > 0 {
            *ptr -= 1;
        }
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

/// Encapsulates iteration and indexing over text.
///
/// This enables us to expose a common interface between a `String` and
/// a `Vec<u32>`. Specifically, a `Vec<u32>` is used for lexical renaming.
trait Text {
    /// An iterator over characters.
    ///
    /// Must be reversible.
    type IdxChars: Iterator + DoubleEndedIterator;

    /// The length of the text.
    fn len(&self) -> u32;

    /// The character previous to the byte index `i`.
    fn prev(&self, i: u32) -> (u32, u32);

    /// The character at byte index `i`.
    fn char_at(&self, i: u32) -> u32;

    /// An iterator over characters tagged with their byte offsets.
    fn char_indices(&self) -> Self::IdxChars;

    /// Compare two strings at byte indices `w1` and `w2`.
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
        let str::CharRange { ch: c, next: j } =
            self.s.char_range_at_reverse(i as usize);
        // This is slower, but "stable."
        // let (j, c) = self.s[..i as usize].char_indices().rev().next().unwrap();
        (j as u32, c as u32)
    }

    #[inline]
    fn char_at(&self, i: u32) -> u32 {
        self.s.char_at(i as usize) as u32
        // This seems slower, but "stable."
        // self.s[i as usize..].chars().next().unwrap() as u32
    }

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

/// A trait for converting indexed characters into a uniform representation.
trait IdxChar {
    /// Convert `Self` to a `(usize, u32)`.
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
