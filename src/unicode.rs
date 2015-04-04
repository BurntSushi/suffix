/// Mask of the value bits of a continuation byte
const CONT_MASK: u8 = 0b0011_1111;
/// Value of the tag bits (tag mask is !CONT_MASK) of a continuation byte
const TAG_CONT_U8: u8 = 0b1000_0000;

// https://tools.ietf.org/html/rfc3629
static UTF8_CHAR_WIDTH: [u8; 256] = [
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, // 0x1F
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, // 0x3F
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, // 0x5F
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, // 0x7F
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // 0x9F
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // 0xBF
0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, // 0xDF
3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3, // 0xEF
4,4,4,4,4,0,0,0,0,0,0,0,0,0,0,0, // 0xFF
];

#[inline(never)]
pub fn char_range_at_reverse(text: &str, start: usize) -> (u32, usize) {
    let mut prev = start;

    prev = prev.saturating_sub(1);
    if text.as_bytes()[prev] < 128 {
        return (text.as_bytes()[prev] as u32, prev)
    }

    // Multibyte case is a fn to allow char_range_at_reverse to inline cleanly
    fn multibyte_char_range_at_reverse(s: &str, mut i: usize) -> (u32, usize) {
        // while there is a previous byte == 10......
        while i > 0 && s.as_bytes()[i] & !CONT_MASK == TAG_CONT_U8 {
            i -= 1;
        }

        let first= s.as_bytes()[i];
        let w = UTF8_CHAR_WIDTH[first as usize];
        assert!(w != 0);

        let mut val = utf8_first_byte(first, w as u32);
        val = utf8_acc_cont_byte(val, s.as_bytes()[i + 1]);
        if w > 2 { val = utf8_acc_cont_byte(val, s.as_bytes()[i + 2]); }
        if w > 3 { val = utf8_acc_cont_byte(val, s.as_bytes()[i + 3]); }
        (val, i)
    }

    return multibyte_char_range_at_reverse(text, prev);
}

/// Return the initial codepoint accumulator for the first byte.
/// The first byte is special, only want bottom 5 bits for width 2, 4 bits
/// for width 3, and 3 bits for width 4.
#[inline]
fn utf8_first_byte(byte: u8, width: u32) -> u32 {
    (byte & (0x7F >> width)) as u32
}

/// Return the value of `ch` updated with continuation byte `byte`.
#[inline]
fn utf8_acc_cont_byte(ch: u32, byte: u8) -> u32 {
    (ch << 6) | (byte & CONT_MASK) as u32
}
