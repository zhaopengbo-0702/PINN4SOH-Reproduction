# Vendored Prefix Relocation Helper

`conda_pack/prefixes.py` is unmodified from the official PyPI `conda-pack==0.8.1` wheel. It is loaded as a standalone module; no Conda manager or package solver is invoked.

- Official artifact: https://files.pythonhosted.org/packages/a2/2b/0e7e7b30b5e8cc5fb5bbed684b9f4a31e04bcf198561b2deef6c2789638e/conda_pack-0.8.1-py2.py3-none-any.whl
- Wheel SHA256: `4ac43dbb5b5a1ee6f10df27f83fd227bfdd8a99cb2acbe6d05f1e5a0a681d307`
- `prefixes.py` SHA256: `88b58df57aa4f6772f8d6aeeb85fe1f91bfd31646ced75560fbe91267a60dc8f`
- Wheel size: 34,228 bytes.
- Upstream license retained in `conda_pack-0.8.1.dist-info/LICENSE.txt`; the prefix module also retains its original conda portability license notices.

The installer runs only on Linux x86_64. The POSIX binary implementation pads NUL bytes at the end of each complete string after replacing the prefix, preserving both suffixes and total binary length. The installer separately limits prefix/shebang length and verifies that recorded placeholders were removed. It preserves original package archive checksums and records before/after checksums for every relocated file.
