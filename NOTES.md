A Rust library for suffix trees and suffix arrays.

It is currently in development/prototype stage. It's a playground for now, but
I intend for it to implement *actually usable* generalized suffix trees and
arrays that are correct and efficient. This is a marked difference from most or
all existing suffix tree implementations that I know of. Namely, most of them
are "research" quality code. By that, I mean that they seem to lack convenient
interfaces (like real generalized suffix trees/arrays), or assume a constant
alphabet (e.g., for computational biology applications) or lack any kind of
Unicode support (which requires careful tweaking from most academic
descriptions of construction algorithms, since they often assume constant
alphabets or dismiss a practioner's concerns of a large integer alphabet).

To build suffix trees, we'll first build a suffix array (with the lengths of
least common prefixes for adjacent suffixes) and use that to construct a suffix
tree. Currently, I believe the algorithm has time complexity `O(n * logm)`
where `n` is the size of the string being indexed and `m` is the size of the
alphabet. It is easy enough to remove the `logm` factor by using a hash map,
but this probably incurs a bit of overhead and loses lexicographic ordering of
nodes in the tree. (Thus, we just use the standard library `BTree`.)

One possible saving grace here is that the "size of the alphabet" actually
means "the size of the alphabet in the string." My guess is that in most cases,
this alphabet will be extremely small relative to the full alphabet actually
supported (i.e., every Unicode character).

The above algorithm has a prototype implementation already. Suffix array
construction is done naively. My plan is to implement the algorithm described
in [(Nong et al.,
2009)](https://local.ugene.unipro.ru/tracker/secure/attachment/12144/Linear%20Suffix%20Array%20Construction%20by%20Almost%20Pure%20Induced-Sorting.pdf),
which is a fast linear time construction by using induced sorting. I've found
the paper to be nearly impenetrable, but
[(Shrestha et al.,
2014)](http://bib.oxfordjournals.org/content/15/2/138.full.pdf) provide a more
accessible description of the algorithm that I think I understand. There also
exists a linear time construction of the LCP array via induced sorting that is
described in [(Fischer, 2011)](http://arxiv.org/pdf/1101.3448.pdf).

OK, I've implemented the algorithm from Nong et al. in `src/array.rs`.
QuickCheck says it's equivalent to the naive implementation, which gives me
some confidence that it is actually correct. But there is still a lot of work
to be done because it's performing way more allocation than is necessary. It
also has a reliance on a sentinel that I'd like to remove. Although, it may be
useful when I adapt this to generalized suffix arrays. Moreover, using the
sentinel can be provably correct because we can use an invalid Unicode scalar
value.

Clearly, I am still in the "how do I implement fast construction" algorithm
phase. I haven't given much thought yet to a public API and what kinds of
operations we should expose. Certainly, it would be easier to do suffix array
-> suffix tree, and define all operations there. But this requires clients to
pay the memory overhead of a suffix tree. Thankfully,
[(Abouelhoda et al.,
2003)](http://ac.els-cdn.com/S1570866703000650/1-s2.0-S1570866703000650-main.pdf?_tid=6661b2e4-8d1e-11e4-ad52-00000aab0f01&acdnat=1419612423_f0a8ad1f2d7a5a98389b796f95d356e0)
describe how to implement many suffix tree algorithms with suffix arrays. I
have yet to thoroughly review this paper, so it isn't yet clear to me if this
is a "optimal time complexity" paper or a "optimal practitioner's algorithm"
kind of a paper. Using suffix arrays directly will undoubtedly be more complex.

I'm also looking for ideas on testing. My instinct is to capture the invariants
of suffix trees/arrays in QuickCheck properties, and use those for randomized
testing. Thankfully, generation is easy in this case: any arbitrary Unicode
string will do.

