# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/lstasiak/emblema/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                                      |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------------------------------------------ | -------: | -------: | -------: | -------: | ------: | --------: |
| src/emblema/catalog/adapters/archive/block\_corpus\_archive.py                            |       76 |        0 |        8 |        0 |    100% |           |
| src/emblema/catalog/adapters/in\_memory/corpus\_reader.py                                 |       29 |        0 |        6 |        0 |    100% |           |
| src/emblema/catalog/adapters/in\_memory/corpus\_repository.py                             |       18 |        0 |        4 |        0 |    100% |           |
| src/emblema/catalog/adapters/persistence/corpus\_record.py                                |       21 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/adapters/persistence/corpus\_repository.py                            |       33 |        1 |        4 |        1 |     95% |        46 |
| src/emblema/catalog/adapters/persistence/corpus\_version\_record.py                       |       45 |        0 |        2 |        0 |    100% |           |
| src/emblema/catalog/adapters/persistence/orm.py                                           |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/adapters/readers/cmapss.py                                            |      107 |        0 |       42 |        0 |    100% |           |
| src/emblema/catalog/adapters/readers/skab.py                                              |      122 |        0 |       44 |        0 |    100% |           |
| src/emblema/catalog/adapters/readers/smd.py                                               |       90 |        0 |       30 |        0 |    100% |           |
| src/emblema/catalog/adapters/readers/subsets.py                                           |        9 |        0 |        4 |        0 |    100% |           |
| src/emblema/catalog/adapters/synthetic/dials.py                                           |        6 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/adapters/synthetic/draws.py                                           |       16 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/adapters/synthetic/latent\_factor\_process.py                         |       26 |        0 |        2 |        0 |    100% |           |
| src/emblema/catalog/adapters/synthetic/layouts.py                                         |        9 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/adapters/synthetic/sensor\_layout.py                                  |       29 |        0 |        4 |        0 |    100% |           |
| src/emblema/catalog/adapters/synthetic/synthetic\_corpus\_reader.py                       |      108 |        0 |       14 |        0 |    100% |           |
| src/emblema/catalog/adapters/tokenisation/sliding\_window.py                              |      115 |        0 |       32 |        0 |    100% |           |
| src/emblema/catalog/application/assemblers/corpus\_version\_ref\_assembler.py             |        6 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/application/assemblers/published\_corpus\_manifest\_assembler.py      |       28 |        0 |        2 |        0 |    100% |           |
| src/emblema/catalog/application/use\_cases/publish\_corpus.py                             |       39 |        0 |        4 |        0 |    100% |           |
| src/emblema/catalog/application/use\_cases/register\_corpus.py                            |       16 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/application/use\_cases/register\_corpus\_version.py                   |       33 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/application/use\_cases/tokenise\_corpus\_version.py                   |       52 |        0 |        4 |        0 |    100% |           |
| src/emblema/catalog/contracts/corpus\_version\_ref.py                                     |       24 |        0 |        8 |        0 |    100% |           |
| src/emblema/catalog/contracts/events.py                                                   |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/contracts/exceptions.py                                               |        8 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/contracts/identifiers.py                                              |        4 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/contracts/published\_channel.py                                       |        7 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/contracts/published\_channel\_statistics.py                           |        3 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/contracts/published\_corpus\_manifest.py                              |       29 |        0 |       16 |        0 |    100% |           |
| src/emblema/catalog/contracts/published\_corpus\_manifest\_json.py                        |       98 |        0 |       22 |        0 |    100% |           |
| src/emblema/catalog/domain/channels/channel\_schema.py                                    |       27 |        0 |        6 |        0 |    100% |           |
| src/emblema/catalog/domain/channels/channel\_statistics.py                                |       19 |        0 |        6 |        0 |    100% |           |
| src/emblema/catalog/domain/channels/channel\_vocabulary.py                                |       48 |        0 |       24 |        0 |    100% |           |
| src/emblema/catalog/domain/exceptions.py                                                  |       84 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/domain/identifiers.py                                                 |       12 |        0 |        2 |        0 |    100% |           |
| src/emblema/catalog/domain/measurements/corpus\_unit.py                                   |       14 |        0 |        2 |        0 |    100% |           |
| src/emblema/catalog/domain/measurements/observation.py                                    |       12 |        0 |        6 |        0 |    100% |           |
| src/emblema/catalog/domain/measurements/static\_feature.py                                |       10 |        0 |        4 |        0 |    100% |           |
| src/emblema/catalog/domain/measurements/time\_extent.py                                   |       15 |        0 |        4 |        0 |    100% |           |
| src/emblema/catalog/domain/registry/corpus.py                                             |       54 |        0 |       20 |        0 |    100% |           |
| src/emblema/catalog/domain/registry/corpus\_content.py                                    |       10 |        0 |        4 |        0 |    100% |           |
| src/emblema/catalog/domain/registry/corpus\_description.py                                |        6 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/domain/registry/corpus\_source.py                                     |        9 |        0 |        4 |        0 |    100% |           |
| src/emblema/catalog/domain/registry/corpus\_version.py                                    |       51 |        0 |       14 |        0 |    100% |           |
| src/emblema/catalog/domain/registry/licence.py                                            |       10 |        0 |        4 |        0 |    100% |           |
| src/emblema/catalog/domain/tokenisation/archived\_corpus.py                               |       15 |        0 |        8 |        0 |    100% |           |
| src/emblema/catalog/domain/tokenisation/placed\_window.py                                 |        6 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/domain/tokenisation/tokenisation\_manifest.py                         |       36 |        0 |        8 |        0 |    100% |           |
| src/emblema/catalog/domain/tokenisation/tokenisation\_scheme.py                           |       46 |        0 |       10 |        0 |    100% |           |
| src/emblema/catalog/domain/tokenisation/unit\_split.py                                    |       26 |        0 |       10 |        0 |    100% |           |
| src/emblema/catalog/domain/tokenisation/window\_reconstruction.py                         |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/domain/tokenisation/window\_spec.py                                   |       21 |        0 |        6 |        0 |    100% |           |
| src/emblema/catalog/ports/corpus\_archive.py                                              |       13 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/ports/corpus\_reader.py                                               |       10 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/ports/corpus\_repository.py                                           |        7 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/ports/tokeniser.py                                                    |       11 |        0 |        0 |        0 |    100% |           |
| src/emblema/config/artifact\_store\_settings.py                                           |        8 |        0 |        0 |        0 |    100% |           |
| src/emblema/config/compute\_tiers.py                                                      |       34 |        0 |        2 |        0 |    100% |           |
| src/emblema/config/database\_settings.py                                                  |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/config/settings.py                                                            |       18 |        0 |        2 |        0 |    100% |           |
| src/emblema/entrypoints/cli/configured.py                                                 |       13 |        0 |        2 |        0 |    100% |           |
| src/emblema/entrypoints/cli/pretrain/adapters.py                                          |       11 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/cli/pretrain/composition\_root.py                                 |       55 |        0 |        4 |        0 |    100% |           |
| src/emblema/entrypoints/cli/pretrain/pretrain\_cli.py                                     |       87 |        0 |       16 |        0 |    100% |           |
| src/emblema/entrypoints/cli/pretrain/pretrain\_invocation.py                              |       11 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/cli/pretrain/services.py                                          |        7 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/cli/pretrain/source\_revision.py                                  |       59 |        0 |       14 |        1 |     99% |   88-\>92 |
| src/emblema/entrypoints/cli/publish\_corpus/adapters.py                                   |        9 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/cli/publish\_corpus/composition\_root.py                          |       64 |        0 |       10 |        0 |    100% |           |
| src/emblema/entrypoints/cli/publish\_corpus/known\_corpora.py                             |       19 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/cli/publish\_corpus/publish\_corpus\_cli.py                       |       37 |        0 |        2 |        0 |    100% |           |
| src/emblema/entrypoints/cli/publish\_corpus/publish\_corpus\_invocation.py                |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/cli/publish\_corpus/services.py                                   |        7 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/adapters/blocks/block\_training\_corpus\_reader.py                |       44 |        0 |        2 |        0 |    100% |           |
| src/emblema/pretraining/adapters/diagnostics/channel\_series.py                           |       24 |        0 |        2 |        0 |    100% |           |
| src/emblema/pretraining/adapters/diagnostics/cross\_channel\_ridge\_baseline.py           |       47 |        0 |       10 |        0 |    100% |           |
| src/emblema/pretraining/adapters/diagnostics/linear\_interpolation\_baseline.py           |       19 |        0 |        6 |        0 |    100% |           |
| src/emblema/pretraining/adapters/diagnostics/own\_and\_cross\_channel\_ridge\_baseline.py |       63 |        0 |       16 |        1 |     99% | 129-\>127 |
| src/emblema/pretraining/adapters/diagnostics/ridge\_regression.py                         |       24 |        0 |       10 |        0 |    100% |           |
| src/emblema/pretraining/adapters/diagnostics/spectral\_recovery.py                        |       57 |        0 |       10 |        0 |    100% |           |
| src/emblema/pretraining/adapters/diagnostics/triviality\_diagnostic.py                    |       55 |        0 |       14 |        0 |    100% |           |
| src/emblema/pretraining/adapters/diagnostics/unit\_bootstrap.py                           |       39 |        0 |        8 |        0 |    100% |           |
| src/emblema/pretraining/adapters/diagnostics/window\_arrays.py                            |       19 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/adapters/documents/experiment\_configuration\_document.py         |       17 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/adapters/documents/fields.py                                      |       67 |        0 |       14 |        0 |    100% |           |
| src/emblema/pretraining/adapters/documents/pretraining\_order\_json.py                    |       30 |        0 |        4 |        0 |    100% |           |
| src/emblema/pretraining/adapters/documents/pretraining\_result\_json.py                   |       40 |        0 |        4 |        0 |    100% |           |
| src/emblema/pretraining/adapters/encoder/encoder\_block.py                                |       15 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/adapters/encoder/fourier\_time\_encoding.py                       |       15 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/adapters/encoder/learned\_channel\_embedding.py                   |        8 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/adapters/encoder/masked\_mean\_pooling.py                         |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/adapters/encoder/self\_attention.py                               |       21 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/adapters/encoder/set\_encoder.py                                  |       24 |        0 |        2 |        0 |    100% |           |
| src/emblema/pretraining/adapters/encoder/tier\_architecture.py                            |        4 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/adapters/experiments/experiment\_file.py                          |       42 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/adapters/handoff/artifact\_store\_handoff\_exchange.py            |       19 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/adapters/handoff/handoff\_training\_runtime.py                    |       19 |        0 |        2 |        0 |    100% |           |
| src/emblema/pretraining/adapters/in\_memory/backbone\_repository.py                       |       13 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/adapters/in\_memory/experiment\_tracker.py                        |       27 |        0 |        6 |        0 |    100% |           |
| src/emblema/pretraining/adapters/in\_memory/handoff\_exchange.py                          |       33 |        0 |        4 |        0 |    100% |           |
| src/emblema/pretraining/adapters/in\_memory/training\_corpus\_reader.py                   |       18 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/adapters/in\_memory/training\_runtime.py                          |       54 |        0 |       12 |        0 |    100% |           |
| src/emblema/pretraining/adapters/mlflow/mlflow\_experiment\_tracker.py                    |       37 |        0 |        8 |        0 |    100% |           |
| src/emblema/pretraining/adapters/objective/masked\_reconstruction.py                      |       15 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/adapters/objective/reconstruction\_decoder.py                     |       20 |        0 |        4 |        0 |    100% |           |
| src/emblema/pretraining/adapters/objective/reconstruction\_loss.py                        |       19 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/adapters/objective/token\_masking.py                              |       35 |        0 |        2 |        0 |    100% |           |
| src/emblema/pretraining/adapters/objective/token\_masks.py                                |       18 |        0 |        4 |        0 |    100% |           |
| src/emblema/pretraining/adapters/persistence/backbone\_record.py                          |       53 |        0 |        2 |        0 |    100% |           |
| src/emblema/pretraining/adapters/persistence/backbone\_repository.py                      |       18 |        0 |        2 |        0 |    100% |           |
| src/emblema/pretraining/adapters/persistence/orm.py                                       |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/adapters/persistence/pretraining\_input\_record.py                |       28 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/adapters/training/device\_generator.py                            |       18 |        6 |       10 |        3 |     54% |20, 22, 29-32 |
| src/emblema/pretraining/adapters/training/devices.py                                      |        5 |        0 |        2 |        0 |    100% |           |
| src/emblema/pretraining/adapters/training/exceptions.py                                   |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/adapters/training/torch\_precision.py                             |       26 |        0 |        4 |        0 |    100% |           |
| src/emblema/pretraining/adapters/training/torch\_training\_runtime.py                     |      152 |        0 |       32 |        2 |     99% |261-\>263, 280-\>279 |
| src/emblema/pretraining/adapters/training/trained\_model.py                               |       33 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/adapters/training/training\_checkpoint.py                         |       36 |        0 |       10 |        0 |    100% |           |
| src/emblema/pretraining/application/use\_cases/accept\_pretraining\_result.py             |       29 |        0 |        2 |        0 |    100% |           |
| src/emblema/pretraining/application/use\_cases/assess\_reconstruction\_run.py             |       11 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/application/use\_cases/fulfil\_pretraining\_order.py              |       21 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/application/use\_cases/order\_pretraining.py                      |       32 |        0 |        2 |        0 |    100% |           |
| src/emblema/pretraining/application/use\_cases/pretrain\_backbone.py                      |       28 |        0 |        4 |        0 |    100% |           |
| src/emblema/pretraining/domain/assessment/assessment.py                                   |       31 |        0 |        8 |        0 |    100% |           |
| src/emblema/pretraining/domain/assessment/check.py                                        |       42 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/domain/assessment/curve.py                                        |        3 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/domain/assessment/decision.py                                     |       34 |        0 |       10 |        0 |    100% |           |
| src/emblema/pretraining/domain/assessment/interval.py                                     |       18 |        0 |        4 |        0 |    100% |           |
| src/emblema/pretraining/domain/assessment/kind\_summary.py                                |       20 |        0 |        2 |        0 |    100% |           |
| src/emblema/pretraining/domain/assessment/mask\_kind\_tally.py                            |       12 |        0 |        4 |        0 |    100% |           |
| src/emblema/pretraining/domain/assessment/results.py                                      |       16 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/domain/assessment/rules.py                                        |      116 |        0 |       24 |        0 |    100% |           |
| src/emblema/pretraining/domain/assessment/spectrum.py                                     |       12 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/domain/assessment/summarised\_run.py                              |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/domain/backbone/backbone.py                                       |       41 |        0 |       10 |        0 |    100% |           |
| src/emblema/pretraining/domain/backbone/pretraining\_input.py                             |       12 |        0 |        4 |        0 |    100% |           |
| src/emblema/pretraining/domain/encoder\_architecture.py                                   |       29 |        0 |        8 |        0 |    100% |           |
| src/emblema/pretraining/domain/exceptions.py                                              |       44 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/domain/handoff/pretraining\_order.py                              |       25 |        0 |        8 |        0 |    100% |           |
| src/emblema/pretraining/domain/handoff/pretraining\_result.py                             |       43 |        0 |       16 |        0 |    100% |           |
| src/emblema/pretraining/domain/identifiers.py                                             |        4 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/domain/learning\_rate\_schedule.py                                |       21 |        0 |       10 |        0 |    100% |           |
| src/emblema/pretraining/domain/mask\_kind.py                                              |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/domain/masking\_strategy.py                                       |       18 |        0 |       10 |        0 |    100% |           |
| src/emblema/pretraining/domain/training/checkpoint\_policy.py                             |       11 |        0 |        4 |        0 |    100% |           |
| src/emblema/pretraining/domain/training/epoch\_outcome.py                                 |       16 |        0 |        8 |        0 |    100% |           |
| src/emblema/pretraining/domain/training/experiment\_configuration.py                      |       26 |        0 |        6 |        0 |    100% |           |
| src/emblema/pretraining/domain/training/precision.py                                      |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/domain/training/run\_position.py                                  |       16 |        0 |        4 |        0 |    100% |           |
| src/emblema/pretraining/domain/training/run\_signature.py                                 |       22 |        0 |        2 |        0 |    100% |           |
| src/emblema/pretraining/domain/training/training\_budget.py                               |       32 |        2 |       16 |        2 |     92% |    91, 93 |
| src/emblema/pretraining/domain/training/training\_corpus.py                               |       12 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/domain/training/training\_corpus\_shape.py                        |       13 |        0 |        8 |        0 |    100% |           |
| src/emblema/pretraining/domain/training/training\_outcome.py                              |       20 |        0 |        8 |        0 |    100% |           |
| src/emblema/pretraining/ports/backbone\_repository.py                                     |        6 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/ports/experiment\_tracker.py                                      |        8 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/ports/handoff\_exchange.py                                        |        9 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/ports/mask\_kind\_summariser.py                                   |        6 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/ports/training\_corpus\_reader.py                                 |        7 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/ports/training\_runtime.py                                        |        8 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/arrays/token\_batch.py                                        |       38 |        0 |        4 |        0 |    100% |           |
| src/emblema/shared/adapters/in\_memory/artifact\_store.py                                 |       28 |        0 |        6 |        0 |    100% |           |
| src/emblema/shared/adapters/in\_memory/clock.py                                           |        6 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/in\_memory/event\_publisher.py                                |        8 |        0 |        2 |        0 |    100% |           |
| src/emblema/shared/adapters/in\_memory/event\_subscriber.py                               |       10 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/in\_memory/id\_generator.py                                   |        9 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/loaders/seeded\_shuffle\_sampler.py                           |       18 |        0 |        2 |        0 |    100% |           |
| src/emblema/shared/adapters/loaders/window\_dataset.py                                    |       12 |        0 |        2 |        0 |    100% |           |
| src/emblema/shared/adapters/loaders/window\_loader.py                                     |       20 |        0 |        2 |        0 |    100% |           |
| src/emblema/shared/adapters/persistence/naming.py                                         |        2 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/storage/files.py                                              |       29 |        0 |        6 |        0 |    100% |           |
| src/emblema/shared/adapters/storage/layout.py                                             |       11 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/storage/local\_directory.py                                   |       55 |        0 |       10 |        0 |    100% |           |
| src/emblema/shared/adapters/storage/s3.py                                                 |       65 |        1 |       14 |        2 |     96% |80-\>85, 111 |
| src/emblema/shared/adapters/system/clock.py                                               |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/system/id\_generator.py                                       |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/tensors/token\_tensors.py                                     |       30 |        0 |        2 |        0 |    100% |           |
| src/emblema/shared/adapters/windows/exceptions.py                                         |        9 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/windows/format.py                                             |       29 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/windows/window\_block.py                                      |       77 |        0 |       12 |        0 |    100% |           |
| src/emblema/shared/adapters/windows/window\_block\_writer.py                              |      118 |        0 |       26 |        1 |     99% | 251-\>255 |
| src/emblema/shared/events/domain\_event.py                                                |        8 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/kernel/artifacts.py                                                    |        8 |        0 |        2 |        0 |    100% |           |
| src/emblema/shared/kernel/checksums.py                                                    |       42 |        0 |        8 |        0 |    100% |           |
| src/emblema/shared/kernel/compute.py                                                      |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/kernel/exceptions.py                                                   |       10 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/kernel/identifiers.py                                                  |       17 |        0 |        2 |        0 |    100% |           |
| src/emblema/shared/kernel/ordering.py                                                     |       10 |        0 |        2 |        0 |    100% |           |
| src/emblema/shared/kernel/sampling.py                                                     |        4 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/kernel/timestamps.py                                                   |       11 |        0 |        4 |        0 |    100% |           |
| src/emblema/shared/kernel/tokens.py                                                       |       47 |        0 |       20 |        0 |    100% |           |
| src/emblema/shared/ports/artifact\_store.py                                               |       12 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/ports/clock.py                                                         |        3 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/ports/event\_publisher.py                                              |        3 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/ports/event\_subscriber.py                                             |        4 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/ports/exceptions.py                                                    |        3 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/ports/id\_generator.py                                                 |        3 |        0 |        0 |        0 |    100% |           |
| **TOTAL**                                                                                 | **5290** |   **10** |  **960** |   **13** | **99%** |           |

60 empty files skipped.


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/lstasiak/emblema/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/lstasiak/emblema/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/lstasiak/emblema/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/lstasiak/emblema/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Flstasiak%2Femblema%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/lstasiak/emblema/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.