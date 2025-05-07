#!/usr/bin/env python
"""
Test runner for Portfolio Allocation System

This script runs all unit and integration tests for the portfolio allocation system.
To run a specific test suite, use the -s/--suite argument:
    python run_tests.py --suite unit
    python run_tests.py --suite integration
    python run_tests.py --suite all (default)

For verbose output, add the -v/--verbose flag:
    python run_tests.py --verbose
"""
import argparse
import unittest
import os
import sys
import importlib.util

def import_module_from_file(file_path):
    """Import a module from a file path."""
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def discover_tests(directory):
    """Discover all test files in a directory."""
    test_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.startswith('test_') and file.endswith('.py'):
                test_files.append(os.path.join(root, file))
    return test_files

def run_test_suite(suite_name, verbose=False):
    """Run a test suite by name."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define suite paths
    suite_paths = {
        'unit': os.path.join(base_dir, 'unit'),
        'integration': os.path.join(base_dir, 'integration'),
    }
    
    # Create test suite
    loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    if suite_name == 'all':
        # Run all test suites
        for suite in suite_paths.values():
            if os.path.exists(suite):
                test_files = discover_tests(suite)
                for test_file in test_files:
                    try:
                        module = import_module_from_file(test_file)
                        test_suite.addTests(loader.loadTestsFromModule(module))
                    except Exception as e:
                        print(f"Error loading tests from {test_file}: {e}")
    else:
        # Run specific suite
        if suite_name not in suite_paths:
            print(f"Error: Unknown test suite '{suite_name}'")
            return False
        
        suite_path = suite_paths[suite_name]
        if not os.path.exists(suite_path):
            print(f"Error: Test suite directory not found: {suite_path}")
            return False
        
        test_files = discover_tests(suite_path)
        for test_file in test_files:
            try:
                module = import_module_from_file(test_file)
                test_suite.addTests(loader.loadTestsFromModule(module))
            except Exception as e:
                print(f"Error loading tests from {test_file}: {e}")
    
    # Run tests
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

def main():
    parser = argparse.ArgumentParser(description='Portfolio Allocation System Test Runner')
    parser.add_argument('-s', '--suite', choices=['all', 'unit', 'integration'], 
                        default='all', help='Test suite to run (default: all)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    print(f"Running {args.suite} tests {'with verbose output' if args.verbose else ''}...")
    success = run_test_suite(args.suite, args.verbose)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()