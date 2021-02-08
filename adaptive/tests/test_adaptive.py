"""
Unit and regression test for the adaptive package.
"""

# Import package, test suite, and other packages as needed
import adaptive
import pytest
import sys

def test_adaptive_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "adaptive" in sys.modules
