# Custom Einops `rearrange` Implementation for NumPy

Submitted by: Deepak, AI Junior @ NITK, Surathkal <br>
Linkedin: [Profile](https://www.linkedin.com/in/deepakcnayak/)

## Approach

The implementation largely is inspired by the original implementation of `rearrange` function in the `einops` library.
The original implementation can handle multiple backends (NumPy, PyTorch, TensorFlow), but this version is specifically tailored for NumPy arrays. This implementation also supports the functionality of `repeat` which is not available in the original implementation of `rearrange`.

Drawing from the original implementation, the core logic operates in two main phases:

1.  **Recipe Preparation (`_prepare_rearrange_recipe`):**
    *   **Parsing:** The input `pattern` string is parsed using the `ParsedExpression` class to identify left/right sides, axis names, parentheses `()`, anonymous dimensions `1`, and ellipsis `...`. It determines the structural "composition" of axes.
    *   **Analysis:** It analyzes the parsed structure against the input tensor's rank (`ndim`). It resolves ellipsis, identifies axes to be kept, dropped, repeated, or inserted. It determines the sequence of required NumPy operations.
    *   **Caching:** The result of this analysis (an immutable `RearrangeRecipe` object) is cached using `functools.lru_cache`. This avoids redundant parsing and analysis for repeated calls with the same pattern signature (pattern, ndim, axes lengths keys).

2.  **Execution (`rearrange`):**
    *   **Recipe Retrieval:** Fetches the cached `RearrangeRecipe` if available, or prepares a new one if not.
    *   **Size Inference:** Calculates the concrete size of each axis based on the input `tensor.shape` and any provided `**axes_lengths`. It validates shape compatibility during this step.
    *   **NumPy Operations:** Executes the plan defined in the recipe using efficient NumPy functions in a specific order:
        1.  Optional initial `reshape` (for splitting axes).
        2.  Optional `np.squeeze` (for dropping input dimensions of size 1).
        3.  Optional `np.transpose` (for reordering remaining axes).
        4.  Optional `np.expand_dims` (for inserting new dimensions of size 1).
        5.  Optional `np.broadcast_to` (for repeating axes / implementing newly created axes from a size-1 source).
        6.  Final `reshape` (for merging axes into the desired output structure).
   * Recipe optimization ensures many common tasks (e.g., transpositions, splits, merges, or repeats) often require just **one** NumPy operation at runtime.

## Supported Operations 

*   **Reshaping:** Achieved implicitly through the combination of other operations (splitting, merging, dropping, inserting, transposing) determined by the pattern.
*   **Transposition:** Handled by `np.transpose` using a permutation calculated during recipe preparation. 
*   **Splitting Axes:** Implemented via `np.reshape`. Parentheses on the left side of the pattern (e.g., `(h w)`) signal splitting, requiring necessary axis lengths in `**axes_lengths` to determine the target shape for this reshape.
*   **Merging Axes:**  Also implemented via the `np.reshape`. Parentheses on the right side (e.g., `-> (h w)`) signal merging. The sizes inferred or provided for the elementary axes within the parentheses are multiplied to calculate the size of the merged dimension.
*   **Repeating Axes:** Handles creating new axes on the right side (e.g., `a 1 c -> a b c`) like in the `repeat` function of the `einops` library. This is done by broadcasting the size-1 source axis to the new size, which is inferred or provided in `**axes_lengths`. However , thiis implementation does not support all the features of the `repeat` function. For example, it does not support repeating multiple axes at once or repeating an axis with a size greater than 1.

## Parsing (`ParsedExpression`)

*   **Input/Output Axes:** Identifies named axes (identifiers) and anonymous axes ('1') on both sides of the `->`.
*   **Ellipsis Recognition:** Detects `...` and replaces it internally for easier processing during analysis and expansion based on `ndim`.
*   **Elementary Operations:** Parses parentheses `()` to understand grouping (for splitting/merging) and spaces/`1` to understand dimension structure. It doesn't directly parse "operations" but rather the structure from which the operations are derived in `_prepare_rearrange_recipe`.

## Error Handling

*   A custom `EinopsError` subclass of `ValueError` is used for specific, informative messages.
*   Checks are implemented for:
    *   **Invalid Patterns:** Syntax errors (mismatched parens, invalid chars/identifiers, incorrect ellipsis, missing '->').
    *   **Shape Mismatches:** Incompatibility between pattern structure and tensor `ndim`, non-divisible dimensions during splitting, incorrect size products.
    *   **Axis Lengths:** Missing required `**axes_lengths` for splitting/new axes, conflicting provided lengths, non-integer/non-positive lengths.
    *   **Identifier Issues:** Duplicate identifiers, dropping non-'1' axes, creating axes without a size-1 source for repetition.

## Optimization & Design

The code is largely modular, utilises OOPs for parsing, testing, creating recipe instances, and is redable with docstrings and comments explaining the logic. 
* The primary optimization is caching the `RearrangeRecipe` via `functools.lru_cache` in `_prepare_rearrange_recipe`, dramatically speeding up repeated calls. This is inspired by original implementation's caching strategy.
* Most transformations are done in a single step using NumPy's optimized functions, avoiding unnecessary intermediate copies. Most time is spent on the initial parsing and recipe preparation and caching, making sure we use as few NumPy operations as possible. **Dry runs tracing the operations are provided in the end**

## Unit Tests

*   Comprehensive unit tests are provided using Python's `unittest` module. 17 unit tests with with 50+ patterns covered.  
*   Tests cover:
    *   Basic operations (transpose, split, merge, add/drop 1, repeat).
    *   Ellipsis usage.
    *   Edge cases (scalar input, empty arrays, zero-sized dimensions).
    *   Complex combined patterns.
    *   Extensive error condition checking for invalid patterns and shape mismatches.
*   Where available, test results are compared against the reference `einops` library implementation for correctness using `np.testing.assert_array_equal`.\

## How to Run

1.  Simply run the notebook `einops_custom_rearrange.ipynb`. It has the implementation, unit tests and efficiency benchmarking.
2. Clone the repo, install the dependencies and run the file `test_rearrange.py` to run the tests. Example on how to use the module below:
   ```python
   import numpy as np
   from custom_rearrange_implementation import rearrange, EinopsError

   # Example:
   x = np.random.rand(1, 4, 5)
   y = rearrange(x, 'a b c -> a b c 1')

   print(y)
   
   ```

# Benchmarking
![comparison between custom and original implementations](./assets/comparison_plot.png)

Although the custom implementation is not as fast as the original one, it performs competitively for all use cases and handles all edge cases.

# Dry runs - NumPy operations tracing
The following dry runs are based on the patterns and shapes used in the unit tests. The goal is to trace the number of NumPy operations performed for each pattern.
We need to trace which of these conditional blocks are entered and executed within the `rearrange` function body for each example:

1.  **Initial Reshape (`recipe.needs_initial_reshape`)**
2.  **Drop Anon '1's (`recipe.dropped_anon1_indices`)** - uses `np.squeeze`
3.  **Transposition (`recipe.axes_permutation`)** - uses `np.transpose`
4.  **Insert '1's (`recipe.inserted_1_indices`)** - uses `np.expand_dims` (potentially multiple times in a loop)
5.  **Repetition (`repetition_needed`)** - uses `np.broadcast_to`
6.  **Final Reshape (`final_target_shape != current_tensor.shape`)** - uses `np.reshape`

Tracing:

1.  **Transpose (`'h w -> w h'`)**
    *   Input Shape: `(3, 4)`
    *   1. Initial Reshape? No.
    *   2. Squeeze? No.
    *   3. Transpose? Yes (`axes=(1, 0)`). Tensor becomes `(4, 3)`.
    *   4. Expand Dims? No.
    *   5. Broadcast? No.
    *   6. Final Reshape? No (already `(4, 3)`).
    *   **Total NumPy Ops: 1** (`transpose`)

2.  **Split an axis (`'(h w) c -> h w c'`, `h=3`)**
    *   Input Shape: `(12, 10)`
    *   1. Initial Reshape? Yes (`reshape((3, 4, 10))`). Tensor becomes `(3, 4, 10)`.
    *   2. Squeeze? No.
    *   3. Transpose? No (permutation is `(0, 1, 2)`).
    *   4. Expand Dims? No.
    *   5. Broadcast? No.
    *   6. Final Reshape? No (already `(3, 4, 10)`).
    *   **Total NumPy Ops: 1** (`reshape`)

3.  **Merge axes (`'a b c -> (a b) c'`)**
    *   Input Shape: `(3, 4, 5)`
    *   1. Initial Reshape? No.
    *   2. Squeeze? No.
    *   3. Transpose? No (permutation is `(0, 1, 2)`).
    *   4. Expand Dims? No.
    *   5. Broadcast? No.
    *   6. Final Reshape? Yes (`reshape((12, 5))`). Tensor becomes `(12, 5)`.
    *   **Total NumPy Ops: 1** (`reshape`)

4.  **Repeat an axis (`'a 1 c -> a b c'`, `b=4`)**
    *   Input Shape: `(3, 1, 5)`
    *   1. Initial Reshape? No (left side `a 1 c` matches `ndim=3`).
    *   2. Squeeze? No (the `1` is used as source, not dropped).
    *   3. Transpose? No (permutation `(0, 1, 2)`).
    *   4. Expand Dims? No.
    *   5. Broadcast? Yes (`broadcast_to((3, 4, 5))`). Tensor becomes `(3, 4, 5)`.
    *   6. Final Reshape? No (already `(3, 4, 5)`).
    *   **Total NumPy Ops: 1** (`broadcast_to`)

5.  **Handle batch dimensions (`'... h w -> ... (h w)'`)**
    *   Input Shape: `(2, 3, 4, 5)`
    *   1. Initial Reshape? No (ellipsis expands).
    *   2. Squeeze? No.
    *   3. Transpose? No (permutation `(0, 1, 2, 3)`).
    *   4. Expand Dims? No.
    *   5. Broadcast? No.
    *   6. Final Reshape? Yes (`reshape((2, 3, 20))`). Tensor becomes `(2, 3, 20)`.
    *   **Total NumPy Ops: 1** (`reshape`)


# AI usage acknowledgement
Grok 3 and Gemini 2.5 were used for extensive test case generation, documentation and debugging. All AI generated content was reviewed and modified by me to ensure good practices, correctness and clarity. 
