import unittest
import numpy as np
import warnings

from custom_rearrange_implementation import rearrange, EinopsError


try:
    from einops import rearrange as einops_rearrange
    EINOTS_AVAILABLE = True
except ImportError:
    EINOTS_AVAILABLE = False
    warnings.warn("Reference 'einops' library not found. Tests comparing against it will be skipped.", ImportWarning)

skip_if_no_einops = unittest.skipUnless(EINOTS_AVAILABLE, "Reference 'einops' library not installed")

class TestRearrangeImplementationSimplified(unittest.TestCase):
    """
    Unit tests for the custom rearrange implementation, focusing on clarity and core functionality.
    Compares results against the reference einops library where available.
    """

    # --- Helper Assertion Methods ---

    def assert_rearrange_matches_reference(self, pattern, tensor, **axes_lengths):
        """Asserts custom rearrange output matches reference einops output."""
        if not EINOTS_AVAILABLE:
            self.skipTest("Reference 'einops' library not available for comparison.")

        try:
            expected_output = einops_rearrange(tensor, pattern, **axes_lengths)
        except Exception as e_ref:
            # If reference fails, maybe the pattern is invalid for both.
            # We'll test specific error cases separately. Log this and assume error tests cover it.
            print(f"\nNote: Reference einops failed for pattern='{pattern}', shape={tensor.shape}, lengths={axes_lengths}. Error: {e_ref}")
            # Assert our implementation *also* fails for this pattern
            with self.assertRaises((EinopsError, ValueError, TypeError), # Expecting failure
                                   msg=f"Reference failed, but custom didn't raise error for pattern='{pattern}', shape={tensor.shape}"):
                rearrange(tensor, pattern, **axes_lengths)
            return # Skip comparison since reference failed

        try:
            actual_output = rearrange(tensor, pattern, **axes_lengths)
        except Exception as e_cust:
            self.fail(f"Custom rearrange failed unexpectedly for pattern='{pattern}', shape={tensor.shape}, lengths={axes_lengths}. "
                      f"Reference succeeded. Error: {type(e_cust).__name__}: {e_cust}")

        np.testing.assert_array_equal(expected_output, actual_output,
                                      err_msg=f"Output mismatch for pattern='{pattern}', shape={tensor.shape}, lengths={axes_lengths}.")
        self.assertEqual(expected_output.shape, actual_output.shape,
                         f"Shape mismatch for pattern='{pattern}', shape={tensor.shape}, lengths={axes_lengths}.")

    def assert_raises_einops_error(self, pattern, tensor, error_substring=None, **axes_lengths):
        """Asserts that the custom rearrange raises EinopsError, optionally checking message content."""
        with self.assertRaises(EinopsError, msg=f"Expected EinopsError for pattern='{pattern}', shape={tensor.shape}, lengths={axes_lengths}") as cm:
            rearrange(tensor, pattern, **axes_lengths)
        if error_substring:
            self.assertIn(error_substring, str(cm.exception),
                          msg=f"Error message mismatch for pattern='{pattern}'. Expected substring '{error_substring}' not found in '{cm.exception}'")

    # --- Core Functionality Tests ---

    @skip_if_no_einops
    def test_01_transpose(self):
        """Test axis transposition."""
        x = np.random.rand(2, 3, 4)
        self.assert_rearrange_matches_reference('a b c -> c b a', x)
        self.assert_rearrange_matches_reference('a b c -> b a c', x)
        self.assert_rearrange_matches_reference('... c -> c ...', x) # With ellipsis

    @skip_if_no_einops
    def test_02_split_axis(self):
        """Test splitting one axis into multiple axes."""
        x = np.random.rand(12, 10)
        self.assert_rearrange_matches_reference('(h w) c -> h w c', x, h=3) # Infer w=4
        self.assert_rearrange_matches_reference('(h w) c -> w h c', x, w=4) # Infer h=3, test order

        x2 = np.random.rand(5, 6 * 7)
        self.assert_rearrange_matches_reference('a (b c d) -> a b c d', x2, b=6, c=7) # Infer d=1

    @skip_if_no_einops
    def test_03_merge_axes(self):
        """Test merging multiple axes into one."""
        x = np.random.rand(3, 4, 5)
        self.assert_rearrange_matches_reference('a b c -> (a b) c', x)
        self.assert_rearrange_matches_reference('a b c -> a (b c)', x)
        self.assert_rearrange_matches_reference('a b c -> (a b c)', x)

    @skip_if_no_einops
    def test_04_split_merge_combination(self):
        """Test combined splitting and merging."""
        x = np.random.rand(6, 8, 3) # (2*3) (4*2) 3
        self.assert_rearrange_matches_reference('(h1 h2) (w1 w2) c -> h1 w1 (h2 w2 c)', x, h1=2, w1=4)

    # --- Dimension '1' and Repetition Tests ---

    @skip_if_no_einops
    def test_05_handle_dimension_1(self):
        """Test adding, removing, and processing dimensions of size 1."""
        x_no_1 = np.random.rand(5, 6)
        self.assert_rearrange_matches_reference('a b -> a 1 b', x_no_1) # Add '1'

        x_with_1 = np.random.rand(5, 1, 6)
        self.assert_rearrange_matches_reference('a 1 c -> a c', x_with_1) # Drop '1'
        self.assert_rearrange_matches_reference('a 1 c -> c 1 a', x_with_1) # Transpose with '1'

    @skip_if_no_einops
    def test_06_repeat_from_1(self):
        """Test repeating/broadcasting using an input dimension of size 1."""
        from einops import repeat  # Import inside to keep locality

        x = np.random.rand(3, 1, 5)
        # Repeat middle dim 'b=4' times, using the size 1 dimension as source
        expected = repeat(x, 'a 1 c -> a b c', b=4)
        actual = rearrange(x, 'a 1 c -> a b c', b=4)
        np.testing.assert_array_equal(expected, actual,
                                      err_msg="Mismatch in broadcasting repeat for middle dim")
        self.assertEqual(expected.shape, actual.shape)

        x2 = np.random.rand(1, 4, 5)
        # Repeat first dim 'a=2' times
        expected2 = repeat(x2, '1 b c -> a b c', a=2)
        actual2 = rearrange(x2, '1 b c -> a b c', a=2)
        np.testing.assert_array_equal(expected2, actual2,
                                      err_msg="Mismatch in broadcasting repeat for first dim")
        self.assertEqual(expected2.shape, actual2.shape)

    @skip_if_no_einops
    def test_07_ellipsis(self):
        """Test ellipsis (...) for arbitrary dimensions."""
        x = np.random.rand(2, 3, 4, 5, 6)
        self.assert_rearrange_matches_reference('a b ... y z -> z y ... b a', x) # Transpose ends
        self.assert_rearrange_matches_reference('a b c ... -> (a b) c ...', x) # Merge with ellipsis
        self.assert_rearrange_matches_reference('... x y z -> ... z (x y)', x) # Merge with ellipsis
        self.assert_rearrange_matches_reference('(a b) ... -> a b ...', x, a=2) # Split with ellipsis

    # --- Edge Case Tests ---

    @skip_if_no_einops
    def test_08_scalar_input(self):
        """Test handling of scalar (0D) input."""
        x = np.array(5)
        self.assert_rearrange_matches_reference('-> ', x)     # Identity
        self.assert_rearrange_matches_reference('... -> ...', x) # Identity
        self.assert_rearrange_matches_reference('-> 1', x)     # Add dim
        self.assert_rearrange_matches_reference('-> 1 1', x)   # Add multiple dims

    @skip_if_no_einops
    def test_09_empty_input(self):
        """Test handling of tensors with a zero-sized dimension."""
        x0 = np.zeros((0, 3, 4))
        self.assert_rearrange_matches_reference('a b c -> c b a', x0)

        x1 = np.zeros((3, 0, 4))
        self.assert_rearrange_matches_reference('a b c -> (a c) b', x1)

    # --- Complex Patterns ---
    @skip_if_no_einops
    def test_10_complex_pattern_image_patches(self):
        """Test a more complex pattern: image to patches."""
        x_img = np.random.rand(10, 64, 64, 3) # B H W C
        patch_h, patch_w = 16, 16
        # B (GridH PatchH) (GridW PatchW) C -> B (GridH GridW) (PatchH PatchW C)
        self.assert_rearrange_matches_reference(
            'b (h ph) (w pw) c -> b (h w) (ph pw c)', x_img, ph=patch_h, pw=patch_w
        )

    # --- Error Handling Tests ---

    def test_11_error_invalid_syntax(self):
        """Test syntax errors in the pattern."""
        x = np.random.rand(2, 3)
        self.assert_raises_einops_error('a b c', x, "separator")
        self.assert_raises_einops_error('a (b c -> a b c', x, "Mismatched parentheses")
        self.assert_raises_einops_error('a b ) c -> a b c', x, "Mismatched parentheses")
        self.assert_raises_einops_error('a (b (c d)) -> a b c d', np.random.rand(2, 12), "Nested")
        self.assert_raises_einops_error('a () c -> a c', np.random.rand(2, 1, 3), "Empty parentheses")
        self.assert_raises_einops_error('a .. b -> a b', x, "ellipsis")

    def test_12_error_invalid_identifiers(self):
        """Test invalid or duplicate axis names."""
        x = np.random.rand(2, 3)
        self.assert_raises_einops_error('_a b -> b _a', x, "underscore")
        self.assert_raises_einops_error('a 2 -> a', np.random.rand(2, 2), "Numerical axis")
        self.assert_raises_einops_error('a a -> a', np.random.rand(2, 2), "Duplicate identifier") # Left
        self.assert_raises_einops_error('a b -> a a', x, "Duplicate identifier") # Right

    def test_13_error_identifier_mismatch(self):
        """Test missing or unused identifiers between left and right."""
        x = np.random.rand(2, 3, 4)
        self.assert_raises_einops_error('a b c -> a b d', x, "New axis 'd' on right side") # New axis 'd' not defined on left or via lengths
        self.assert_raises_einops_error('a b c -> a b', x, "were specified on the left but are not used on the right") # Drop non-'1' axis 'c'
        self.assert_raises_einops_error('a b -> a b c', np.random.rand(2,3), "requires size") # Introduce 'c' without size

    def test_14_error_shape_incompatibility(self):
        """Test patterns incompatible with tensor shape (e.g., divisibility)."""
        x = np.random.rand(12, 10)
        self.assert_raises_einops_error('(h w) c -> h w c', x, "not divisible", h=5) # 12 not divisible by 5
        self.assert_raises_einops_error('(h w) c -> h w c', x, "Cannot infer sizes") # Both h,w missing

        x3d = np.random.rand(2,3,4)
        self.assert_raises_einops_error('a b -> (a b)', x3d, "implies 2 dimensions, but input tensor has 3") # Wrong ndim

    def test_15_error_axis_length(self):
        """Test errors related to missing, conflicting, or invalid axis lengths."""
        x = np.random.rand(12, 10)
        self.assert_raises_einops_error('(h w) c -> h w c', x, "Cannot infer") # Missing lengths for split
        self.assert_raises_einops_error('a b -> a b c', x, "requires size") # Missing length for new axis

        # Conflicting length for split
        self.assert_raises_einops_error('(h w) c -> h w c', x, "Shape mismatch", h=3, w=5) # 3*5 != 12

        # Invalid length value
        self.assert_raises_einops_error('(h w) c -> h w c', x, "positive int", h=-3)
        self.assert_raises_einops_error('(h w) c -> h w c', x, "positive int", h=0)

    def test_16_error_repetition(self):
        """Test errors during axis repetition."""
        x_no_1 = np.random.rand(3, 2, 5)
        # Cannot introduce new axis 'c' without a size-1 source dimension
        self.assert_raises_einops_error('a b d -> a b c d', x_no_1, "requires an input axis of size 1", c=4)

        x_one_1 = np.random.rand(1, 5)
        # Need two size-1 sources ('a' and 'c'), but only one is available
        self.assert_raises_einops_error('1 b -> a c b', x_one_1, "requires an input axis of size 1", a=2, c=3)

    def test_17_error_scalar_patterns(self):
        """Test invalid patterns for scalar input."""
        x = np.array(5)
        self.assert_raises_einops_error('-> a', x, "New axis 'a' on right side")
        self.assert_raises_einops_error('a -> ', x, "must be empty or '...' for scalar") # Left side must be empty


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
