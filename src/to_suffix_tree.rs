use {SuffixArray, SuffixTree, Node, Rawlink, array_naive};

pub fn to_suffix_tree<'a, 's>(sa: &'a SuffixArray<'s>) -> SuffixTree<'s> {
    let mut st = SuffixTree::init(&*sa.text);
    let mut last: *mut Node = &mut *st.root;
    for (i, &sufstart) in sa.indices.iter().enumerate() {
        let lcp_len = sa.lcp_lens[i];
        let vins = ancestor_lcp_len(unsafe { &mut *last }, lcp_len);
        let dv = vins.root_concat_len();
        if dv == lcp_len {
            // The concatenation of the labels from root-to-vins equals
            // the longest common prefix of SA[i-1] and SA[i].
            // This means that the suffix we're adding contains the
            // entirety of `vins`, which in turn means we can simply
            // add it as a new leaf.

            let mut node = box Node::new(sf!(sa.text, sufstart + lcp_len));
            let first_char = node.key();

            // TODO: I don't yet understand why this invariant is true,
            // but it has to be---otherwise the algorithm is flawed. ---AG
            assert!(!vins.children.contains_key(&first_char));

            last = &mut *node;
            node.parent = Rawlink::some(vins);
            node.terminal = true;
            vins.children.insert(first_char, node);
        } else if dv < lcp_len {
            // In this case, `vins`'s right-most child overlaps with the
            // suffix we're trying to insert. So we need to:
            //   1) Cut the right-most edge (but keep the node).
            //   2) Create a new internal node whose path-label is the LCP.
            //   3) Attach the old node to the new internal node. Its
            //      label should be what it was before, but with the LCP
            //      removed.
            //   4) Add a new leaf to this internal node containing the
            //      current suffix with the LCP removed.

            // Why is this invariant true?
            // Well, the only way this can't be true is if `vins` is
            // `last`, which is the last leaf that was added. (If `vins`
            // isn't `last`, then `vins` must be a parent of `last`, which
            // implies it has a child.)
            // But we also know that the current suffix is > than the last
            // suffix inserted, which means the lcp of the last suffix and
            // this suffix can be at *most* len(last). Therefore, when
            // `vins` is `last`, we have that `len(last) >= len(lcp)`,
            // which implies that `len(last)` (== `dv`) can never be less
            // than `len(lcp)` (== `lcp_len`).
            assert!(vins.children.len() > 0);
            // Thus, we can pick the right-most child with impunity.
            let rkey = *vins.children.keys().next_back().unwrap();

            // 1) cut the right-most edge
            let mut rnode = vins.children.remove(&rkey).unwrap();

            // 2) create new internal node (full path label == LCP)
            let mut int_node = box Node::new(
                s!(sa.text, sa.indices[i-1] + dv, sa.indices[i-1] + lcp_len));
            int_node.parent = Rawlink::some(vins);
            int_node.terminal = false;

            // 3) Attach old node to new internal node and update
            // the label.
            rnode.label =
                s!(sa.text,
                   sa.indices[i-1] + lcp_len,
                   sa.indices[i-1] + rnode.root_concat_len());
            rnode.parent = Rawlink::some(&mut *int_node);

            // 4) Create new leaf node with the current suffix, but with
            // the lcp trimmed.
            let mut leaf = box Node::new(sf!(sa.text, sufstart + lcp_len));
            last = &mut *leaf;
            leaf.parent = Rawlink::some(&mut *int_node);
            leaf.terminal = true;

            // Finally, attach all of the nodes together.
            assert!(rnode.key() != leaf.key()); // why is this true? ---AG
            int_node.children.insert(rnode.key(), rnode);
            int_node.children.insert(leaf.key(), leaf);
            vins.children.insert(int_node.key(), int_node);
        } else {
            unreachable!()
        }
    }
    st
}

fn ancestor_lcp_len<'a, 's>(start_node: &'a mut Node<'s>, lcp_len: u32)
                           -> &'a mut Node<'s> {
    // Is it worth making a mutable `Ancestors` iterator?
    // If this is the only place that needs it, probably not. ---AG
    let mut cur = start_node;
    loop {
        // FIXME: root_concat_len traverses all the way to the root!
        // So I think that makes this function `O(logn * logn)`. Owch.
        // We need to store the full path label length I think.
        if cur.root_concat_len() <= lcp_len {
            return cur;
        }
        match cur.parent.resolve() {
            // We've reached the root, so we have no choice but to use it.
            None => break,
            Some(p) => { cur = p; }
        }
    }
    cur
}

#[cfg(test)]
mod tests {
    use quickcheck::quickcheck;
    use {Node, array_naive};

    #[test]
    fn basic() {
        let sa = array_naive("banana");
        let st = sa.to_suffix_tree();
    }

    #[test]
    fn basic2() {
        let sa = array_naive("apple");
        let st = sa.to_suffix_tree();
    }

    #[test]
    fn basic3() {
        let sa = array_naive("mississippi");
        let st = sa.to_suffix_tree();
    }

    #[test]
    fn qc_n_leaves() {
        fn prop(s: String) -> bool {
            let sa = array_naive(&*s);
            let st = sa.to_suffix_tree();
            st.root.leaves().count() == s.len() + 1
        }
        quickcheck(prop as fn(String) -> bool);
    }

    #[test]
    fn qc_internals_have_at_least_two_children() {
        fn prop(s: String) -> bool {
            let sa = array_naive(&*s);
            let st = sa.to_suffix_tree();
            for node in st.root.preorder() {
                if !node.is_terminal() && node.children.len() < 2 {
                    return false;
                }
            }
            true
        }
        quickcheck(prop as fn(String) -> bool);
    }

    #[test]
    fn qc_tree_enumerates_suffixes() {
        fn prop(s: String) -> bool {
            // This is pretty much relying on `array_naive` to produce the
            // correct suffixes. But the nice thing about the naive algorithm
            // is that it's stupidly simple.
            let sa = array_naive(&*s);
            let st = sa.to_suffix_tree();
            for (i, leaf) in st.root.leaves().enumerate() {
                let mut ancestors: Vec<&Node> = leaf.ancestors().collect();
                ancestors.reverse();

                let suffix: String =
                    ancestors.iter().map(|node| node.label).collect();
                if suffix != sa.suffix(i) {
                    return false;
                }
            }
            true
        }
    }

    #[test]
    fn scratch() {
        let sa = array_naive("mississippi");
        let st = sa.to_suffix_tree();
        debug!("{}", st);
        // let node = st.root.children.get(&'a').unwrap()
                          // .children.get(&'n').unwrap()
                          // .children.get(&'n').unwrap();
        // debug!("{}", st);
        // debug!("NODE: {}", node);
        for n in st.root.leaves() {
            debug!("{}", n.label);
        }
        // for ancestor in node.ancestors().skip(1) {
            // debug!("ancestor: {}", ancestor.label);
        // }
    }
}
