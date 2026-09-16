# SKAB sample

The first 60 rows of three experiments of SKAB v0.9, the Skoltech Anomaly Benchmark, copied byte
for byte with their headers: `anomaly-free/anomaly-free.csv`, `other/1.csv` and `valve1/0.csv`.
Three files rather than one because the format varies across them and the reader has to meet all
of it: the anomaly-free recording carries no `anomaly` or `changepoint` column, `other/1.csv` ends
its lines with a line feed alone, and the rest end them with a carriage return and a line feed.
The rows already hold gaps — the sample spans 62 to 64 seconds — so the nominal one-second cadence
is not something a test can assume.

Source of record: `https://github.com/waico/SKAB/archive/b2c0d46c2971dcbfe71e26087b6d231998bb91c2.zip`
(5,419,930 bytes, SHA-256 `45ac11b460e495ba2c1301c3f8e871b688b5ecfa6cd0770b4225594fe45efc80`),
members under `SKAB-<commit>/data/`. Reference: I. D. Katser and V. O. Kozitsin, "Skoltech Anomaly
Benchmark (SKAB)", Kaggle, 2020. The repository is licensed GPL-3.0 and the data files sit inside
it, so a derivative may be redistributed under the same terms.

This is test data for the corpus reader. Keep the files byte-exact, line endings included: the
reader's checksum is computed over the raw bytes.

Regenerate from the unpacked archive:

```sh
for file in anomaly-free/anomaly-free.csv other/1.csv valve1/0.csv; do
  head -n 61 "SKAB-<commit>/data/$file" > "tests/data/skab/$file"
done
```
