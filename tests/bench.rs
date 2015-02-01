#![feature(test)]

extern crate suffix;
extern crate test;

use suffix::SuffixTable;
use test::Bencher;

#[bench]
fn naive_small(b: &mut Bencher) {
    let s = "mississippi";
    b.iter(|| { SuffixTable::new_naive(s); })
}

#[bench]
fn sais_small(b: &mut Bencher) {
    let s = "mississippi";
    b.iter(|| { SuffixTable::new(s); })
}

#[bench]
fn naive_dna_small(b: &mut Bencher) {
    let s = include_str!("AP009048_10000.fasta");
    b.iter(|| { SuffixTable::new_naive(s); })
}

#[bench]
fn sais_dna_small(b: &mut Bencher) {
    let s = include_str!("AP009048_10000.fasta");
    b.iter(|| { SuffixTable::new(s); })
}

#[bench]
fn naive_dna_medium(b: &mut Bencher) {
    let s = include_str!("AP009048_100000.fasta");
    b.iter(|| { SuffixTable::new_naive(s); })
}

#[bench]
fn sais_dna_medium(b: &mut Bencher) {
    let s = include_str!("AP009048_100000.fasta");
    b.iter(|| { SuffixTable::new(s); })
}

#[bench]
fn search_scan_not_exists(b: &mut Bencher) {
    let s = include_str!("AP009048_100000.fasta");
    b.iter(|| { s.contains("H"); });
}

#[bench]
fn search_suffix_not_exists(b: &mut Bencher) {
    let s = include_str!("AP009048_100000.fasta");
    let st = SuffixTable::new(s);
    b.iter(|| { st.positions("H"); });
}

#[bench]
fn search_suffix_not_exists_contains(b: &mut Bencher) {
    let s = include_str!("AP009048_100000.fasta");
    let st = SuffixTable::new(s);
    b.iter(|| { st.contains("H"); });
}

#[bench]
fn search_scan_exists_one(b: &mut Bencher) {
    let s = include_str!("AP009048_100000.fasta");
    b.iter(|| { s.contains("C"); });
}

#[bench]
fn search_suffix_exists_one(b: &mut Bencher) {
    let s = include_str!("AP009048_100000.fasta");
    let st = SuffixTable::new(s);
    b.iter(|| { st.positions("C"); });
}

#[bench]
fn search_suffix_exists_one_contains(b: &mut Bencher) {
    let s = include_str!("AP009048_100000.fasta");
    let st = SuffixTable::new(s);
    b.iter(|| { st.contains("C"); });
}

#[bench]
fn search_scan_exists_many(b: &mut Bencher) {
    let s = include_str!("AP009048_100000.fasta");
    b.iter(|| { s.contains("ACTTACGTGTCTGC"); });
}

#[bench]
fn search_suffix_exists_many(b: &mut Bencher) {
    let s = include_str!("AP009048_100000.fasta");
    let st = SuffixTable::new(s);
    b.iter(|| { st.positions("ACTTACGTGTCTGC"); });
}

#[bench]
fn search_suffix_exists_many_contains(b: &mut Bencher) {
    let s = include_str!("AP009048_100000.fasta");
    let st = SuffixTable::new(s);
    b.iter(|| { st.contains("ACTTACGTGTCTGC"); });
}
