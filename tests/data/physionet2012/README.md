# PhysioNet/CinC Challenge 2012 sample

Four ICU stays of the PhysioNet/Computing in Cardiology Challenge 2012, copied byte for byte:
`set-a/132539.txt`, `set-a/132548.txt`, `set-a/140501.txt` and `set-b/149509.txt`. Four rather
than one because the format varies across them and the reader has to meet all of it: 132539 has
neither height nor weight recorded (`-1`); in 132548 measurements taken at admission sit between
the descriptors, so the descriptors are not the first six rows; 140501 holds descriptors and not
one measurement, one of three such stays in set A; 149509 records the weight twice at `00:00`, as
the descriptor and as a measurement. Every file ends its lines with a line feed alone.

Source of record: `https://physionet.org/files/challenge-2012/1.0.0/`, archives `set-a.zip`
(7,938,449 bytes, SHA-256 `1433638192fda622e8bb11a4559b8af2dbb1c69f0e1c3b5b606cc12aabd0a4c5`) and
`set-b.zip` (7,958,979 bytes, SHA-256
`54094c39631797af554ad4233164e004a699521351bd2f2bed50d6564ebc56dc`), members
`set-a/<RecordID>.txt` and `set-b/<RecordID>.txt`. Reference: I. Silva, G. Moody, D. J. Scott,
L. A. Celi and R. G. Mark, "Predicting In-Hospital Mortality of ICU Patients: The
PhysioNet/Computing in Cardiology Challenge 2012", Computing in Cardiology 39:245–248, 2012. The
data is published under the Open Data Commons Attribution License v1.0, which permits a
redistributed derivative with attribution.

This is test data for the corpus reader. Keep the files byte-exact: the reader's checksum is
computed over the raw bytes.

Regenerate from the unpacked archives:

```sh
for stay in set-a/132539 set-a/132548 set-a/140501 set-b/149509; do
  cp "data/raw/physionet2012/$stay.txt" "tests/data/physionet2012/$stay.txt"
done
```
