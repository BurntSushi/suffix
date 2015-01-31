use std::fmt;

use {SuffixTable, SuffixTree, vec_from_elem};
use tree::to_suffix_tree;

#[derive(Clone)]
pub struct SuffixArray<'s> {
    table: SuffixTable<'s>,
    inverse: Vec<u32>,
    lcp_lens: Vec<u32>,
}

impl<'s> SuffixArray<'s> {
    pub fn new(text: &'s str) -> SuffixArray<'s> {
        SuffixTable::new(text).into_suffix_array()
    }

    pub fn new_naive(text: &'s str) -> SuffixArray<'s> {
        SuffixTable::new_naive(text).into_suffix_array()
    }

    pub fn from_table(table: SuffixTable<'s>) -> SuffixArray<'s> {
        let mut inverse = vec_from_elem(table.len(), 0u32);
        for (rank, &sufstart) in table.table().iter().enumerate() {
            inverse[sufstart as usize] = rank as u32;
        }
        let lcp_lens = lcp_lens_linear(table.text(), table.table(), &*inverse);
        SuffixArray {
            table: table,
            inverse: inverse,
            lcp_lens: lcp_lens,
        }
    }

    pub fn to_suffix_tree(&self) -> SuffixTree<'s> {
        to_suffix_tree(self)
    }

    pub fn table(&self) -> &[u32] { self.table.table() }

    pub fn text(&self) -> &'s str { self.table.text() }

    #[inline]
    pub fn len(&self) -> usize { self.table.len() }

    #[inline]
    pub fn suffix(&self, i: usize) -> &str { self.table.suffix(i) }

    #[inline]
    pub fn lcp(&self, i: usize) -> &str {
        let sufi = self.table()[i] as usize;
        &self.text()[sufi..sufi + (self.lcp_lens[i] as usize)]
    }

    #[inline]
    pub fn lcp_len(&self, i: usize) -> usize {
        self.lcp_lens[i] as usize
    }
}

impl<'s> fmt::Debug for SuffixArray<'s> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(writeln!(f, "\n-----------------------------------------"));
        try!(writeln!(f, "SUFFIX ARRAY"));
        try!(writeln!(f, "text: {}", self.text()));
        for (rank, &sufstart) in self.table().iter().enumerate() {
            try!(writeln!(f, "suffix[{}] {}, {}",
                          rank, sufstart, self.suffix(rank)));
        }
        for (sufstart, &rank) in self.inverse.iter().enumerate() {
            try!(writeln!(f, "inverse[{}] {}, {}",
                          sufstart, rank, self.suffix(rank as usize)));
        }
        for (i, &len) in self.lcp_lens.iter().enumerate() {
            try!(writeln!(f, "lcp_length[{}] {}", i, len));
        }
        writeln!(f, "-----------------------------------------")
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
            len -= 1;
        }
    }
    lcps
}

#[allow(dead_code)]
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
    a.chars().zip(b.chars()).take_while(|&(ca, cb)| ca == cb).count() as u32
}
