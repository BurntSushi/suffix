use std::cmp;
use std::fmt;
use std::iter::AdditiveIterator;

use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use suffix::SuffixTable;

#[derive(Clone, Debug)]
pub struct DB {
    idx: SuffixTable<'static>,
    documents: Vec<Document>,
}

#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct Document {
    name: String,
    start: usize,
    end: usize,
}

#[derive(Debug)]
pub struct SearchResult<'i> {
    document: &'i Document,
    position: usize,
    line: &'i str,
}

impl Decodable for DB {
    fn decode<D: Decoder>(d: &mut D) -> Result<DB, D::Error> {
        let (table, offsets, texts): (Vec<u32>, Vec<Document>, String) =
            try!(Decodable::decode(d));
        Ok(DB {
            idx: SuffixTable::from_parts(texts, table),
            documents: offsets,
        })
    }
}

impl Encodable for DB {
    fn encode<E: Encoder>(&self, e: &mut E) -> Result<(), E::Error> {
        (self.idx.table(), &self.documents, self.idx.text()).encode(e)
    }
}

impl Decodable for Document {
    fn decode<D: Decoder>(d: &mut D) -> Result<Document, D::Error> {
        let (name, start, end): (String, usize, usize) =
            try!(Decodable::decode(d));
        Ok(Document {
            name: name,
            start: start,
            end: end,
        })
    }
}

impl Encodable for Document {
    fn encode<E: Encoder>(&self, e: &mut E) -> Result<(), E::Error> {
        (&self.name, self.start, self.end).encode(e)
    }
}

impl DB {
    pub fn create(documents: Vec<(String, String)>) -> DB {
        let mut texts = String::with_capacity(
            documents.iter().map(|s| s.1.len()).sum());

        let mut offsets = Vec::with_capacity(documents.len());
        for (name, text) in documents {
            let start = texts.len();
            texts.push_str(&text);
            offsets.push(Document {
                name: name,
                start: start,
                end: texts.len(),
            });
            texts.push('\x00');
        }
        DB {
            idx: SuffixTable::new(texts),
            documents: offsets,
        }
    }

    pub fn search(&self, query: &str) -> Vec<SearchResult> {
        let mut results = Vec::with_capacity(64);
        for i in self.idx.positions(query).iter().map(|&i| i as usize) {
            results.push(self.new_result(query, i));
        }
        results
    }

    pub fn document(&self, position: usize) -> Option<(&Document, usize)> {
        // TODO: Change this to binary search. ---AG
        for d in &self.documents {
            if position >= d.start && position < d.end {
                return Some((d, position - d.start));
            }
        }
        None
    }

    pub fn document_text(&self, doc: &Document) -> &str {
        &self.idx.text()[doc.start..doc.end]
    }

    fn new_result(&self, query: &str, position: usize) -> SearchResult {
        let (doc, position) = self.document(position).unwrap();
        SearchResult {
            document: doc,
            position: position,
            line: &self.document_text(doc)[position..position+query.len()],
        }
    }
}

impl<'i> fmt::Display for SearchResult<'i> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}:{}:{}", self.document.name, self.position, self.line)
    }
}

impl<'i> Eq for SearchResult<'i> {}

impl<'i> PartialEq for SearchResult<'i> {
    fn eq<'j>(&self, o: &SearchResult<'j>) -> bool {
        (&self.document.name, self.position) == (&o.document.name, o.position)
    }
}

impl<'i> PartialOrd for SearchResult<'i> {
    fn partial_cmp<'j>(&self, o: &SearchResult<'j>) -> Option<cmp::Ordering> {
        let this = (&self.document.name, self.position);
        this.partial_cmp(&(&o.document.name, o.position))
    }
}

impl<'i> Ord for SearchResult<'i> {
    fn cmp<'j>(&self, o: &SearchResult<'j>) -> cmp::Ordering {
        self.partial_cmp(o).unwrap()
    }
}
