use std::collections::btree_map::{self, BTreeMap};
use std::fmt;
use std::iter;
use std::ptr;

use SuffixArray;

#[doc(hidden)]
pub struct SuffixTree<'s> {
    text: &'s str,
    root: Box<Node>,
}

#[doc(hidden)]
pub struct Node {
    parent: Rawlink<Node>,
    children: BTreeMap<char, Box<Node>>,
    suffixes: Vec<u32>,
    start: u32,
    end: u32,
    path_len: u32,
}

struct Rawlink<T> {
    p: *mut T,
}

impl<T> Copy for Rawlink<T> {}

impl<'s> SuffixTree<'s> {
    fn init(s: &'s str) -> SuffixTree<'s> {
        SuffixTree {
            text: s,
            root: Node::leaf(0, 0, 0),
        }
    }

    pub fn text(&self) -> &'s str {
        self.text
    }

    pub fn root(&self) -> &Node {
        &*self.root
    }

    pub fn label(&self, node: &Node) -> &'s str {
        &self.text[node.start as usize .. node.end as usize]
    }

    pub fn suffix(&self, node: &Node) -> &'s str {
        assert!(node.suffixes.len() > 0);
        &self.text[node.suffixes[0] as usize..]
    }

    fn key(&self, node: &Node) -> char {
        self.label(node).char_at(0)
    }
}

impl Node {
    pub fn children<'t>(&'t self) -> Children<'t> {
        Children { it: self.children.values() }
    }

    pub fn ancestors<'t>(&'t self) -> Ancestors<'t> {
        Ancestors { cur: Some(self) }
    }

    pub fn preorder<'t>(&'t self) -> Preorder<'t> {
        Preorder::new(self)
    }

    pub fn leaves<'t>(&'t self) -> Leaves<'t> {
        Leaves { it: self.preorder() }
    }

    pub fn suffix_indices<'t>(&'t self) -> SuffixTreeIndices<'t> {
        SuffixTreeIndices { it: self.leaves(), node: None, cur_suffix: 0 }
    }

    pub fn len(&self) -> u32 {
        self.end - self.start
    }

    pub fn depth(&self) -> usize {
        self.ancestors().count() - 1
    }
}

impl Node {
    fn leaf(sufstart: u32, start: u32, end: u32) -> Box<Node> {
        Box::new(Node {
            parent: Rawlink::none(),
            children: BTreeMap::new(),
            suffixes: vec![sufstart],
            start: start,
            end: end,
            path_len: 0,
        })
    }

    fn internal(start: u32, end: u32) -> Box<Node> {
        Box::new(Node {
            parent: Rawlink::none(),
            children: BTreeMap::new(),
            suffixes: vec![],
            start: start,
            end: end,
            path_len: 0,
        })
    }

    fn parent(&self) -> Option<&Node> {
        self.parent.resolve()
    }

    fn has_terminals(&self) -> bool {
        self.suffixes.len() > 0
    }

    fn is_root(&self) -> bool {
        self.parent.resolve().is_none()
    }

    fn add_parent(&mut self, node: &mut Node) {
        self.parent = Rawlink::some(node);
        self.path_len = node.path_len + self.len();
    }
}

/// Rawlink is a type like Option<T> but for holding a raw mutable pointer.
impl<T> Rawlink<T> {
    /// Like `Option::None` for Rawlink.
    fn none() -> Rawlink<T> {
        Rawlink{p: ptr::null_mut()}
    }

    /// Like `Option::Some` for Rawlink
    fn some(n: &mut T) -> Rawlink<T> {
        Rawlink{p: n}
    }

    /// Convert the `Rawlink` into an immutable Option value.
    fn resolve<'a>(&self) -> Option<&'a T> {
        unsafe { self.p.as_ref() }
    }

    /// Convert the `Rawlink` into a mutable Option value.
    fn resolve_mut<'a>(&mut self) -> Option<&'a mut T> {
        unsafe { self.p.as_mut() }
    }
}

impl<'s> fmt::Debug for SuffixTree<'s> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fn fmt<'s>(f: &mut fmt::Formatter, st: &SuffixTree<'s>,
                   node: &Node, depth: usize) -> fmt::Result {
            let indent: String = iter::repeat(' ').take(depth * 2).collect();
            if node.is_root() {
                try!(writeln!(f, "ROOT"));
            } else {
                try!(writeln!(f, "{}{}", indent, st.label(node)));
            }
            for child in node.children() {
                try!(fmt(f, st, child, depth + 1));
            }
            Ok(())
        }
        try!(writeln!(f, "\n-----------------------------------------"));
        try!(writeln!(f, "SUFFIX TREE"));
        try!(writeln!(f, "text: {}", self.text));
        try!(fmt(f, self, self.root(), 0));
        writeln!(f, "-----------------------------------------")
    }
}

impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Node {{ start: {}, end: {}, len(children): {}, \
                           terminals: {}, parent? {} }}",
               self.start, self.end, self.children.len(), self.suffixes.len(),
               self.parent().map(|_| "yes").unwrap_or("no"))
    }
}

#[doc(hidden)]
pub struct Ancestors<'t> {
    cur: Option<&'t Node>,
}

impl<'t> Iterator for Ancestors<'t> {
    type Item = &'t Node;

    fn next(&mut self) -> Option<&'t Node> {
        if let Some(node) = self.cur {
            self.cur = node.parent();
            Some(node)
        } else {
            None
        }
    }
}

#[doc(hidden)]
pub struct Children<'t> {
    it: btree_map::Values<'t, char, Box<Node>>,
}

impl<'t> Iterator for Children<'t> {
    type Item = &'t Node;

    fn next(&mut self) -> Option<&'t Node> {
        self.it.next().map(|n| &**n)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<'t> DoubleEndedIterator for Children<'t> {
    fn next_back(&mut self) -> Option<&'t Node> {
        self.it.next_back().map(|n| &**n)
    }
}

impl<'t> ExactSizeIterator for Children<'t> {}

#[doc(hidden)]
pub struct Preorder<'t> {
    stack: Vec<&'t Node>,
}

impl<'t> Preorder<'t> {
    fn new(start: &'t Node) -> Preorder<'t> {
        Preorder { stack: vec![start] }
    }
}

impl<'t> Iterator for Preorder<'t> {
    type Item = &'t Node;

    fn next(&mut self) -> Option<&'t Node> {
        match self.stack.pop() {
            None => None,
            Some(node) => {
                self.stack.extend(node.children().rev());
                Some(node)
            }
        }
    }
}

#[doc(hidden)]
pub struct Leaves<'t> {
    it: Preorder<'t>,
}

impl<'t> Iterator for Leaves<'t> {
    type Item = &'t Node;

    fn next(&mut self) -> Option<&'t Node> {
        for n in self.it.by_ref() {
            if n.len() > 0 && n.has_terminals() {
                return Some(n);
            }
        }
        None
    }
}

#[doc(hidden)]
pub struct SuffixTreeIndices<'t> {
    it: Leaves<'t>,
    node: Option<&'t Node>,
    cur_suffix: u32,
}

impl<'t> Iterator for SuffixTreeIndices<'t> {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        if let Some(node) = self.node {
            if (self.cur_suffix as usize) < node.suffixes.len() {
                self.cur_suffix += 1;
                return Some(node.suffixes[self.cur_suffix as usize - 1]);
            }
            self.node = None;
            self.cur_suffix = 0;
        }
        match self.it.next() {
            None => None,
            Some(leaf) => { self.node = Some(leaf); self.next() }
        }
    }
}

pub fn to_suffix_tree<'s>(sa: &'s SuffixArray<'s>) -> SuffixTree<'s> {
    fn ancestor_lcp_len<'a>(start: &'a mut Node, lcplen: u32) -> &'a mut Node {
        // Is it worth making a mutable `Ancestors` iterator?
        // If this is the only place that needs it, probably not. ---AG
        let mut cur = start;
        loop {
            if cur.path_len <= lcplen {
                return cur;
            }
            match cur.parent.resolve_mut() {
                // We've reached the root, so we have no choice but to use it.
                None => break,
                Some(p) => { cur = p; }
            }
        }
        cur
    }

    let table = sa.table();
    let mut st = SuffixTree::init(sa.text());
    let mut last: *mut Node = &mut *st.root;
    for (i, &sufstart) in sa.table().iter().enumerate() {
        let lcp_len = sa.lcp_len(i) as u32;
        let vins = ancestor_lcp_len(unsafe { &mut *last }, lcp_len);
        let dv = vins.path_len;
        if dv == lcp_len {
            // The concatenation of the labels from root-to-vins equals
            // the longest common prefix of SA[i-1] and SA[i].
            // This means that the suffix we're adding contains the
            // entirety of `vins`, which in turn means we can simply
            // add it as a new leaf.
            let mut node = Node::leaf(
                sufstart, sufstart + lcp_len, sa.len() as u32);
            node.add_parent(vins);

            let first_char = st.key(&*node);
            // TODO: I don't yet understand why this invariant is true,
            // but it has to be---otherwise the algorithm is flawed. ---AG
            assert!(!vins.children.contains_key(&first_char));

            last = &mut *node;
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
            // than `len(lcp)` (== `lcp_len`). Which in turn implies that
            // we can't be here, since `dv < lcp_len`.
            assert!(vins.children.len() > 0);
            // Thus, we can pick the right-most child with impunity.
            let rkey = *vins.children.keys().next_back().unwrap();

            // 1) cut the right-most edge
            let mut rnode = vins.children.remove(&rkey).unwrap();

            // 2) create new internal node (full path label == LCP)
            let mut int_node = Node::internal(table[i-1] + dv,
                                              table[i-1] + lcp_len);
            int_node.add_parent(vins);

            // 3) Attach old node to new internal node and update
            // the label.
            rnode.start = table[i-1] + lcp_len;
            rnode.end = table[i-1] + rnode.path_len;
            rnode.add_parent(&mut *int_node);

            // 4) Create new leaf node with the current suffix, but with
            // the lcp trimmed.
            let mut leaf = Node::leaf(
                sufstart, sufstart + lcp_len, sa.len() as u32);
            leaf.add_parent(&mut *int_node);

            // Update the last node we visited.
            last = &mut *leaf;

            // Finally, attach all of the nodes together.
            assert!(st.key(&*rnode) != st.key(&*leaf)); // why? ---AG
            int_node.children.insert(st.key(&*rnode), rnode);
            int_node.children.insert(st.key(&*leaf), leaf);
            vins.children.insert(st.key(&*int_node), int_node);
        } else {
            unreachable!()
        }
    }
    st
}

#[cfg(test)]
mod tests {
    use quickcheck::quickcheck;
    use SuffixArray;

    #[test]
    fn basic() {
        let sa = SuffixArray::new_naive("banana");
        let _ = sa.to_suffix_tree();
    }

    #[test]
    fn basic2() {
        let sa = SuffixArray::new_naive("apple");
        let _ = sa.to_suffix_tree();
    }

    #[test]
    fn basic3() {
        let sa = SuffixArray::new_naive("mississippi");
        let _ = sa.to_suffix_tree();
    }

    #[test]
    fn qc_n_leaves() {
        fn prop(s: String) -> bool {
            let sa = SuffixArray::new_naive(&*s);
            let st = sa.to_suffix_tree();
            st.root.leaves().count() == s.len()
        }
        quickcheck(prop as fn(String) -> bool);
    }

    #[test]
    fn qc_internals_have_at_least_two_children() {
        fn prop(s: String) -> bool {
            let sa = SuffixArray::new_naive(&*s);
            let st = sa.to_suffix_tree();
            for node in st.root.preorder() {
                if !node.has_terminals() && node.children.len() < 2 {
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
            // This is pretty much relying on `SuffixArray::new_naive` to
            // produce the correct suffixes. But the nice thing about the naive
            // algorithm is that it's stupidly simple.
            let sa = SuffixArray::new_naive(&*s);
            let st = sa.to_suffix_tree();
            for (i, sufi) in st.root.suffix_indices().enumerate() {
                if &st.text[sufi as usize..] != sa.suffix(i) {
                    return false;
                }
            }
            true
        }
        quickcheck(prop as fn(String) -> bool);
    }

    // #[test]
    // fn scratch() {
        // let sa = SuffixArray::new_naive("mississippi");
        // let st = sa.to_suffix_tree();
        // debug!("{:?}", st);
        // let node = st.root.children.get(&'a').unwrap()
                          // .children.get(&'n').unwrap()
                          // .children.get(&'n').unwrap();
        // debug!("{}", st);
        // debug!("NODE: {}", node);
        // for n in st.root.leaves() {
            // debug!("{}", st.label(n));
        // }
        // for ancestor in node.ancestors().skip(1) {
            // debug!("ancestor: {}", st.label(ancestor));
        // }
    // }
}
