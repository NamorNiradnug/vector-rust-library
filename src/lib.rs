/// Calculates sum of two given integeres.
/// Examples:
/// ```rust
/// use vector_rust_library::add;
/// assert_eq!(add(1, 1), 2);
/// ```
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[test]
fn test_add() {
    assert_eq!(add(2, 2), 4);
}
