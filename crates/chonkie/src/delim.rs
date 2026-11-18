//! Delimiter search for the recursive chunker, compiled once per level.
//!
//! Delimiters are rare, so the search jumps between bytes that could start
//! one: SIMD `memchr`, `memchr2` or `memchr3` when there are at most three
//! distinct first bytes, and a 256-entry table when there are more. Whole
//! patterns are only compared where such a byte sits.

/// A set of delimiters, matched leftmost-first: the earliest match wins, and
/// among matches at the same byte the one listed first wins.
#[derive(Clone, Debug)]
pub(crate) struct Delimiters {
    patterns: Vec<Box<[u8]>>,
    firsts: Vec<u8>,
    table: [bool; 256],
}

impl Delimiters {
    pub(crate) fn new<'a>(patterns: impl IntoIterator<Item = &'a [u8]>) -> Self {
        let patterns: Vec<Box<[u8]>> = patterns
            .into_iter()
            .filter(|p| !p.is_empty())
            .map(Box::from)
            .collect();
        let mut table = [false; 256];
        let mut firsts = Vec::new();
        for p in &patterns {
            if !table[p[0] as usize] {
                table[p[0] as usize] = true;
                firsts.push(p[0]);
            }
        }
        Self {
            patterns,
            firsts,
            table,
        }
    }

    /// The first match at or after `from`, as `(start, end)`.
    pub(crate) fn find(&self, hay: &[u8], mut from: usize) -> Option<(usize, usize)> {
        while from < hay.len() {
            let at = from + self.candidate(&hay[from..])?;
            if let Some(p) = self.patterns.iter().find(|p| hay[at..].starts_with(p)) {
                return Some((at, at + p.len()));
            }
            from = at + 1;
        }
        None
    }

    #[inline]
    fn candidate(&self, hay: &[u8]) -> Option<usize> {
        match *self.firsts.as_slice() {
            [] => None,
            [a] => memchr::memchr(a, hay),
            [a, b] => memchr::memchr2(a, b, hay),
            [a, b, c] => memchr::memchr3(a, b, c, hay),
            _ => hay.iter().position(|&b| self.table[b as usize]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Delimiters;

    /// Leftmost-first by brute force: try every pattern, in order, at every byte.
    fn naive(patterns: &[&[u8]], hay: &[u8], from: usize) -> Option<(usize, usize)> {
        (from..hay.len()).find_map(|at| {
            patterns
                .iter()
                .find(|p| !p.is_empty() && hay[at..].starts_with(p))
                .map(|p| (at, at + p.len()))
        })
    }

    fn all(find: impl Fn(usize) -> Option<(usize, usize)>) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let mut from = 0;
        while let Some((s, e)) = find(from) {
            out.push((s, e));
            from = e;
        }
        out
    }

    #[test]
    fn matches_brute_force_for_every_first_byte_path() {
        let sets: [&[&[u8]]; 7] = [
            &[],
            &[b" "],
            &[b"\n\n", b"\n"],
            &[b"\n", b"\n\n"],
            &[b"\r\n", b"\n", b"\r"],
            &[b". ", b"! ", b"? "],
            &[
                b"...",
                b".",
                b"(",
                b")",
                b",",
                "\u{2014}".as_bytes(),
                b"-",
                b"'",
            ],
        ];
        let alphabet: &[u8] = b"ab .!?\n\r(),-'...";
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        for patterns in sets {
            let delims = Delimiters::new(patterns.iter().copied());
            for len in [0, 1, 2, 5, 40, 400] {
                let mut hay: Vec<u8> = (0..len)
                    .map(|_| {
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        alphabet[seed as usize % alphabet.len()]
                    })
                    .collect();
                hay.extend_from_slice("a\u{2014}b".as_bytes());
                assert_eq!(
                    all(|from| delims.find(&hay, from)),
                    all(|from| naive(patterns, &hay, from)),
                    "patterns {patterns:?}, text {:?}",
                    String::from_utf8_lossy(&hay)
                );
            }
        }
    }
}
