# PhysioNet/CinC Challenge 2019 sample

Four ICU stays of the PhysioNet/Computing in Cardiology Challenge 2019, copied byte for byte:
`training_setA/p000001.psv`, `training_setA/p000015.psv`, `training_setA/p000022.psv` and
`training_setB/p100006.psv`. Four rather than one because the format varies across them and the
reader has to meet all of it: p000001 is a long stay of 54 hours in no recorded ward (`Unit1` and
`Unit2` both `NaN`); p000015 is a short stay that develops sepsis, so ten of its rows carry the
label the reader must check and never read as a value; p000022 is flagged as in the surgical ICU
and its hours start at 5, not 1; p100006 is a stay of set B whose hours start at 6 and whose
admission offset is exactly zero. Every file ends its lines with a line feed alone.

Source of record: `https://physionet.org/files/challenge-2019/1.0.0/training/`, directories
`training_setA/` (20,336 stays) and `training_setB/` (20,000 stays), one pipe-separated file per
stay. Reference: M. A. Reyna, C. S. Josef, R. Jeter, S. P. Shashikumar, M. B. Westover, S. Nemati,
G. D. Clifford and A. Sharma, "Early Prediction of Sepsis From Clinical Data: The
PhysioNet/Computing in Cardiology Challenge 2019", Critical Care Medicine 48(2):210–217, 2020. The
data is published under the Creative Commons Attribution 4.0 International licence, which permits
a redistributed derivative with attribution.

This is test data for the corpus reader. Keep the files byte-exact: the reader's checksum is
computed over the raw bytes.

Regenerate from the downloaded files:

```sh
for stay in training_setA/p000001 training_setA/p000015 training_setA/p000022 training_setB/p100006; do
  cp "data/raw/physionet2019/training/$stay.psv" "tests/data/physionet2019/$stay.psv"
done
```
