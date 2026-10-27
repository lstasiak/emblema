# eICU Collaborative Research Database demo sample

Five unit stays of the eICU Collaborative Research Database demo, every row of each in the four
tables the reader reads, as the database published them: `patient.csv.gz`, `nurseCharting.csv.gz`,
`lab.csv.gz` and `intakeOutput.csv.gz`. The rows are byte for byte the database's, cut back out of
the parser's lines, so a quoted field holding a comma or a line break stays whole.

| Stay | Hospital | Why it is here |
|---|---|---|
| `258915`, `301185` | 93 | two stays of one hospital, each with 30 of the variables read; the first has no gender recorded |
| `1817131` | 268 | a stay of another hospital, with 28 variables |
| `376041` | 120 | a patient older than 89, whom the database records as `> 89`; its 18 variables are measured by the others too, so it can be held out |
| `3035602` | 420 | a stay of a surgical unit without a single reading, which is still a unit |

Source of record: `https://physionet.org/content/eicu-crd-demo/2.0.1/`, tables
`patient.csv.gz` (136,791 bytes, SHA-256
`15e1eb52169828e3cfff3b67132a087298de4d0365bbb31bb294f6043bffd474`), `nurseCharting.csv.gz`
(16,838,725 bytes, `475000d63d4ef883b2e891d7197997201300bedc13fcf23e67add1abee11dafe`),
`lab.csv.gz` (5,729,763 bytes, `e06852850abeb65211bc22ac43f17121ba41b14656fac862396449af84d8273a`)
and `intakeOutput.csv.gz` (1,826,646 bytes,
`45db8ee0e64a4e3745284bde5828f740175770c45d00ea0c27daff789d18106e`). Reference: A. Johnson,
T. Pollard, O. Badawi and J. Raffa, "eICU Collaborative Research Database Demo" (version 2.0.1),
PhysioNet, 2021, doi:10.13026/4mxk-na84. The demo is published under the Open Database License
v1.0; this sample is an extract of it and is offered under the same licence, with this
attribution.

This is test data for the corpus reader. Keep the files byte-exact: the reader's checksum is
computed over the raw bytes, and `make_sample.py` compresses each table without a timestamp, so a
rerun on the same tables gives the same bytes.

    uv run python tests/data/eicu_demo/make_sample.py data/raw/eicu_demo tests/data/eicu_demo

The sample holds no repeated reading; 34 of its numbers were taken before the stay began on the
unit, and are not the stay's. The reader's tests build repeats, and every other rule, by hand.
