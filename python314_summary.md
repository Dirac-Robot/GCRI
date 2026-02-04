# Python 3.14 New Features Summary

## 1. Summary of Important New Features
- Upgraded REPL with syntax highlighting, enhancing the developer experience.
- Introduction of Template String Literals (t-strings) for more convenient and readable inline template strings.
- Official support for Free-Threaded Python, enabling true parallel execution by removing the Global Interpreter Lock (GIL).
- Deferred evaluation of annotations improving performance and flexibility.
- Support for subinterpreters built into the standard library.
- Enhancements to standard libraries such as pathlib (adding copy and move methods), argparse (improved help messages), and unittest improvements.
- Experimental Just-In-Time (JIT) compiler improvements that boost performance.
- Addition of support for zstd compression.

## 2. Release Date
- Official release date: October 7, 2025.
- Python 3.14.0 is the initial stable release; subsequent bugfix releases will follow approximately every two months.

## 3. Notable Breaking Changes
- Changes in error message content may cause test breakages (e.g., lexer test_bad_date).
- Migration to free-threaded Python architecture may impact some C extension modules or extensions relying on previous GIL behavior.
- Deprecations and removals typical for a major Python release; refer to official changelog for detailed list.

For full details and the complete changelog, visit the official Python 3.14 release page: https://www.python.org/downloads/release/python-3140/

