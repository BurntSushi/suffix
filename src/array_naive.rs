use SuffixArray;

pub fn construct<'s>(s: &'s str) -> SuffixArray<'s> {
    let mut sufs: Vec<u32> = s.char_indices().map(|(i, _)| i as u32).collect();
    sufs.sort_by(|&a, &b| sf!(s, a).cmp(sf!(s, b)));
    let lcp_lens = suffix_lcp_lens(&*s, &*sufs);
    SuffixArray {
        text: s,
        indices: sufs,
        lcp_lens: lcp_lens,
    }
}

fn suffix_lcp_lens(text: &str, indices: &[u32]) -> Vec<u32> {
    // The first LCP is always 0 because of the definition:
    //   LCP_LENS[i] = lcp_len(suf[i-1], suf[i])
    let mut lcps = Vec::from_elem(indices.len(), 0);
    for (i, win) in indices.windows(2).enumerate() {
        lcps[i + 1] = lcp_len(sf!(text, win[0]), sf!(text, win[1]));
    }
    lcps
}

fn lcp_len(a: &str, b: &str) -> u32 {
    a.chars().zip(b.chars()).take_while(|&(ca, cb)| ca == cb).count() as u32
}

#[cfg(test)]
mod tests {
    use super::construct;

    #[test]
    fn basic() {
        let sa = construct("banana");
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
        let sa = construct("apple");
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
}
