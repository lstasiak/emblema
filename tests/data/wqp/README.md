# Water Quality Portal sample

Six monitoring locations of the US Water Quality Portal, the first 120 result records of each as
the portal published them, in the layout the reader reads: `batches/batch-00000.zip` and
`batches/batch-00001.zip`, each one member `result.csv` of the narrow result profile with every
column kept. The records are byte for byte the portal's, cut back out of the parser's lines, so a
quoted field holding a line break stays whole.

| Batch | Location | Why it is here |
|---|---|---|
| 0 | `USGS-01387610`, `USGS-423412082471600` | locations of the USGS database (`NWIS`); the first repeats characteristics at one instant |
| 0 | `CHESAPEAKEMONITORINGCOOP-CMC.WSR.WASPRU002` | a volunteer programme in the EPA's database (`STORET`), with results taken in air; its three channels are measured by the others too, so it can be held out |
| 0 | `11113300-BCHFSPFRNLF` | enough results, but on characteristics outside the 64 kept here, so it falls below 50 readings |
| 1 | `21VASWCB-9-SNC008.04`, `21VASWCB-1ALIV016.50` | a second batch, read after the first, with quality-control samples, sediment results and repeats at one instant (the second location holds 102 records) |

Source of record: the Water Quality Portal (National Water Quality Monitoring Council, USGS, EPA),
doi:10.5066/P9QRKUVJ, results between 2010-01-01 and 2023-12-31 fetched on 2026-10-26 by
`scripts/wqp_fetch.py` (seed 1, batches of 100 locations). The portal's record in the federal data
catalogue gives the U.S. Public Domain licence; cite the portal with its DOI.

This is test data for the corpus reader. Keep the archives byte-exact: the reader's checksum is
computed over the raw bytes, and `make_sample.py` writes the member with a fixed timestamp, so a
rerun on the same batches gives the same bytes.

    uv run python tests/data/wqp/make_sample.py data/raw/wqp/batches tests/data/wqp/batches

The sample holds no statistic and no result dated alone; the reader's tests build those, and
every other rule, by hand.
