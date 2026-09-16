# SMD sample

The first 60 minutes of two machines of the Server Machine Dataset, one from each of two groups:
`machine-1-1.txt` and `machine-2-1.txt`, copied byte for byte — 60 rows of 38 metrics each.
Two groups rather than one so that a test can select a subset and see the selection matter.

Source of record:
`https://github.com/NetManAIOps/OmniAnomaly/archive/7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4.zip`
(105,339,616 bytes, SHA-256 `5bcce77823a9ed2ea872c733895ccb6a7edb40476306ab8626b49481803b831e`),
members `OmniAnomaly-<commit>/ServerMachineDataset/train/`. Reference: Y. Su, Y. Zhao, C. Niu,
R. Liu, W. Sun and D. Pei, "Robust Anomaly Detection for Multivariate Time Series through
Stochastic Recurrent Neural Network", KDD 2019. The repository is MIT-licensed; the data it
carries was collected from a large Internet company and has no licence statement of its own, so
a redistributed derivative is not assumed to be permitted.

Only training halves are sampled: the test halves carry anomaly labels and are evaluation data
outside the Catalog.

This is test data for the corpus reader. Keep the files byte-exact: the reader's checksum is
computed over the raw bytes.

Regenerate from the unpacked archive:

```sh
for machine in machine-1-1 machine-2-1; do
  head -n 60 "OmniAnomaly-<commit>/ServerMachineDataset/train/$machine.txt" \
    > "tests/data/smd/$machine.txt"
done
```
