#![allow(dead_code)]

use std::fmt;

use SuffixTable;

#[doc(hidden)]
#[derive(Clone)]
pub struct SuffixArray<'s> {
    table: SuffixTable<'s>,
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
        let lcp_lens = table.lcp_lens();
        SuffixArray {
            table: table,
            lcp_lens: lcp_lens,
        }
    }

    pub fn table(&self) -> &[u32] { self.table.table() }

    pub fn text(&self) -> &str { self.table.text() }

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
        for (i, &len) in self.lcp_lens.iter().enumerate() {
            try!(writeln!(f, "lcp_length[{}] {}", i, len));
        }
        writeln!(f, "-----------------------------------------")
    }
}
