# C-MAPSS sample

Two engines from `train_FD001.txt` of NASA's Turbofan Engine Degradation Simulation Data Set
(C-MAPSS): units 39 (128 cycles) and 91 (135 cycles), the two shortest trajectories of the
subset, copied byte for byte — 263 rows, 44,735 bytes.

Source of record:
`https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip`
(12,429,152 bytes, SHA-256 `c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2`),
member `CMAPSSData/train_FD001.txt`. Reference: A. Saxena, K. Goebel, D. Simon and N. Eklund,
"Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM08. The data set
is a work of the U.S. Government; the NASA Prognostics Center of Excellence asks publications to
acknowledge it.

This is test data for the corpus reader. Keep the file byte-exact: every row ends with two
spaces and a line feed, and the reader's checksum is computed over the raw bytes.

Regenerate from the unpacked archive:

```sh
awk '$1 == 39 || $1 == 91' CMAPSSData/train_FD001.txt > tests/data/cmapss/train_FD001.txt
```
