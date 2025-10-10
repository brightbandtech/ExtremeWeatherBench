# Future Cleanup Tasks

This document tracks files that were NOT modified in the `feature/new-event-types` branch but need similar cleanup work for consistency and quality. These should be addressed in separate PRs to avoid scope creep.

## Low Test Coverage Files

These files have low test coverage and would benefit from additional tests:

### Critical Priority (< 20% coverage)
- **`src/extremeweatherbench/calc.py`** (15% coverage)
  - Contains core calculation functions
  - Functions like wind speed calculations need comprehensive tests
  - Should add tests for edge cases and error conditions

### Medium Priority (20-45% coverage)
- **`src/extremeweatherbench/evaluate_cli.py`** (28% coverage)
  - CLI entry points need integration tests
  - Test argument parsing and validation
  - Test error handling for invalid configurations

- **`src/extremeweatherbench/regions.py`** (44% coverage)
  - Regional masking functions need more coverage
  - Test various geographic regions
  - Test edge cases (poles, date line, etc.)

- **`src/extremeweatherbench/cases.py`** (59% coverage)
  - Case building and operator logic
  - Test case validation
  - Test edge cases in time ranges

### Lower Priority (60-70% coverage)
- **`src/extremeweatherbench/defaults.py`** (68% coverage)
  - Mostly constants, but validation functions need tests
  - Test OUTPUT_COLUMNS schema enforcement

## Docstring & Comment Consistency

While the modified files in this branch have been cleaned up, the following unmodified files should receive similar treatment:

### Files needing docstring review:
1. **`src/extremeweatherbench/calc.py`**
   - Review function docstrings for completeness
   - Ensure all parameters documented
   - Check 88-character limit compliance

2. **`src/extremeweatherbench/cases.py`**
   - Review class and method docstrings
   - Ensure consistent formatting
   - Update any stale documentation

3. **`src/extremeweatherbench/regions.py`**
   - Add examples to docstrings where appropriate
   - Document return types clearly

4. **`src/extremeweatherbench/defaults.py`**
   - Document constants and their usage
   - Add module-level docstring if missing

5. **`src/extremeweatherbench/evaluate_cli.py`**
   - Review CLI documentation
   - Ensure help text is clear and concise

## Code Quality Improvements

### General recommendations for all files:
- Run comprehensive linting (ruff, mypy)
- Check for unused imports
- Ensure consistent comment capitalization
- Verify 88-character line limit for comments/docstrings
- Review type hints for completeness

## Testing Strategy

For the files listed above, focus on:
1. **Unit tests** for individual functions
2. **Integration tests** for workflows
3. **Edge case testing** (empty data, boundary conditions)
4. **Error handling tests** (invalid inputs, missing data)

## Priority Order

Recommended order for addressing these in future PRs:

1. **First PR**: Add tests for `calc.py` (highest impact, low coverage)
2. **Second PR**: Docstring cleanup across all files above
3. **Third PR**: Add tests for `regions.py` and `cases.py`
4. **Fourth PR**: CLI testing for `evaluate_cli.py`
5. **Fifth PR**: General code quality pass (linting, formatting)

## Notes

- These items were identified during the PR cleanup for `feature/new-event-types`
- Addressing them separately prevents scope creep in the current PR
- Each PR should focus on one file or one type of improvement for easier review
- Consider creating issues in the project tracker for each item

