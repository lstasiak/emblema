# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/lstasiak/emblema/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                                      |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------------------------------------------ | -------: | -------: | -------: | -------: | ------: | --------: |
| src/emblema/catalog/adapters/archive/block\_corpus\_archive.py                            |       71 |        0 |        6 |        0 |    100% |           |
| src/emblema/catalog/adapters/in\_memory/corpus\_reader.py                                 |       29 |        0 |        6 |        0 |    100% |           |
| src/emblema/catalog/adapters/in\_memory/corpus\_repository.py                             |       18 |        0 |        4 |        0 |    100% |           |
| src/emblema/catalog/adapters/persistence/corpus\_record.py                                |       21 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/adapters/persistence/corpus\_repository.py                            |       33 |        1 |        4 |        1 |     95% |        46 |
| src/emblema/catalog/adapters/persistence/corpus\_version\_record.py                       |       45 |        0 |        2 |        0 |    100% |           |
| src/emblema/catalog/adapters/persistence/orm.py                                           |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/adapters/readers/cmapss.py                                            |      129 |        0 |       44 |        0 |    100% |           |
| src/emblema/catalog/adapters/readers/esa\_ad.py                                           |      169 |        0 |       44 |        0 |    100% |           |
| src/emblema/catalog/adapters/readers/physionet2012.py                                     |      148 |        0 |       56 |        0 |    100% |           |
| src/emblema/catalog/adapters/readers/skab.py                                              |      116 |        0 |       38 |        0 |    100% |           |
| src/emblema/catalog/adapters/readers/smd.py                                               |       81 |        0 |       28 |        0 |    100% |           |
| src/emblema/catalog/adapters/readers/subsets.py                                           |        9 |        0 |        4 |        0 |    100% |           |
| src/emblema/catalog/adapters/readers/text.py                                              |       21 |        0 |        8 |        0 |    100% |           |
| src/emblema/catalog/adapters/synthetic/synthetic\_corpus\_reader.py                       |       91 |        0 |       12 |        0 |    100% |           |
| src/emblema/catalog/adapters/tokenisation/published\_window\_tokeniser.py                 |       34 |        0 |        2 |        0 |    100% |           |
| src/emblema/catalog/adapters/tokenisation/sliding\_window.py                              |      115 |        0 |       32 |        0 |    100% |           |
| src/emblema/catalog/application/assemblers/corpus\_version\_ref\_assembler.py             |        6 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/application/assemblers/published\_corpus\_manifest\_assembler.py      |       31 |        0 |        2 |        0 |    100% |           |
| src/emblema/catalog/application/use\_cases/publish\_corpus.py                             |       40 |        0 |        4 |        0 |    100% |           |
| src/emblema/catalog/application/use\_cases/register\_corpus.py                            |       16 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/application/use\_cases/register\_corpus\_version.py                   |       33 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/application/use\_cases/tokenise\_corpus\_version.py                   |       53 |        0 |        4 |        0 |    100% |           |
| src/emblema/catalog/contracts/corpus\_version\_ref.py                                     |       24 |        0 |        8 |        0 |    100% |           |
| src/emblema/catalog/contracts/events.py                                                   |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/contracts/exceptions.py                                               |       10 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/contracts/identifiers.py                                              |        4 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/contracts/observed\_value.py                                          |       12 |        0 |        6 |        0 |    100% |           |
| src/emblema/catalog/contracts/observed\_window.py                                         |       29 |        0 |       12 |        0 |    100% |           |
| src/emblema/catalog/contracts/published\_channel.py                                       |        7 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/contracts/published\_channel\_statistics.py                           |        3 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/contracts/published\_corpus\_manifest.py                              |       29 |        0 |       16 |        0 |    100% |           |
| src/emblema/catalog/contracts/published\_corpus\_manifest\_json.py                        |      105 |        0 |       26 |        0 |    100% |           |
| src/emblema/catalog/contracts/static\_value.py                                            |       10 |        0 |        4 |        0 |    100% |           |
| src/emblema/catalog/contracts/window\_tokeniser.py                                        |        7 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/domain/channels/channel\_schema.py                                    |       27 |        0 |        6 |        0 |    100% |           |
| src/emblema/catalog/domain/channels/channel\_statistics.py                                |       19 |        0 |        6 |        0 |    100% |           |
| src/emblema/catalog/domain/channels/channel\_vocabulary.py                                |       48 |        0 |       24 |        0 |    100% |           |
| src/emblema/catalog/domain/exceptions.py                                                  |       84 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/domain/identifiers.py                                                 |       31 |        0 |        8 |        0 |    100% |           |
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
| src/emblema/catalog/domain/tokenisation/split\_policy.py                                  |       36 |        0 |        4 |        0 |    100% |           |
| src/emblema/catalog/domain/tokenisation/tokenisation\_manifest.py                         |       36 |        0 |        8 |        0 |    100% |           |
| src/emblema/catalog/domain/tokenisation/tokenisation\_scheme.py                           |       46 |        0 |       10 |        0 |    100% |           |
| src/emblema/catalog/domain/tokenisation/unit\_split.py                                    |       38 |        0 |       16 |        0 |    100% |           |
| src/emblema/catalog/domain/tokenisation/window\_reconstruction.py                         |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/domain/tokenisation/window\_spec.py                                   |       21 |        0 |        6 |        0 |    100% |           |
| src/emblema/catalog/ports/corpus\_archive.py                                              |       13 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/ports/corpus\_reader.py                                               |       10 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/ports/corpus\_repository.py                                           |        7 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/ports/tokeniser.py                                                    |       11 |        0 |        0 |        0 |    100% |           |
| src/emblema/config/api\_settings.py                                                       |       27 |        0 |        0 |        0 |    100% |           |
| src/emblema/config/artifact\_store\_settings.py                                           |        8 |        0 |        0 |        0 |    100% |           |
| src/emblema/config/boosting\_settings.py                                                  |        2 |        0 |        0 |        0 |    100% |           |
| src/emblema/config/broker\_settings.py                                                    |        7 |        0 |        0 |        0 |    100% |           |
| src/emblema/config/compute\_tiers.py                                                      |       34 |        0 |        2 |        0 |    100% |           |
| src/emblema/config/convolution\_settings.py                                               |        3 |        0 |        0 |        0 |    100% |           |
| src/emblema/config/database\_settings.py                                                  |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/config/lora\_settings.py                                                      |        3 |        0 |        0 |        0 |    100% |           |
| src/emblema/config/patch\_settings.py                                                     |        2 |        0 |        0 |        0 |    100% |           |
| src/emblema/config/probe\_settings.py                                                     |        3 |        0 |        0 |        0 |    100% |           |
| src/emblema/config/schedule\_settings.py                                                  |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/config/settings.py                                                            |       42 |        0 |       10 |        0 |    100% |           |
| src/emblema/config/telemetry\_settings.py                                                 |        4 |        0 |        0 |        0 |    100% |           |
| src/emblema/config/worker\_settings.py                                                    |       53 |        0 |       16 |        0 |    100% |           |
| src/emblema/entrypoints/api/adapters.py                                                   |       11 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/api/api\_server.py                                                |       14 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/api/composition\_root.py                                          |       94 |        0 |        8 |        1 |     99% | 239-\>246 |
| src/emblema/entrypoints/api/emblema\_api.py                                               |       42 |        0 |        6 |        0 |    100% |           |
| src/emblema/entrypoints/api/health\_routes.py                                             |       28 |        0 |        2 |        0 |    100% |           |
| src/emblema/entrypoints/api/problem\_details.py                                           |       51 |        0 |       10 |        1 |     98% |  99-\>104 |
| src/emblema/entrypoints/api/readiness.py                                                  |       18 |        0 |        2 |        0 |    100% |           |
| src/emblema/entrypoints/api/readiness\_report.py                                          |        4 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/api/request\_size\_limit.py                                       |       31 |        1 |       10 |        2 |     93% |26, 37-\>50 |
| src/emblema/entrypoints/api/schemas/health\_resource.py                                   |        2 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/api/schemas/readiness\_resource.py                                |        7 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/api/services.py                                                   |       13 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/api/telemetry/counted\_answers.py                                 |       17 |        0 |        2 |        1 |     95% |   25-\>27 |
| src/emblema/entrypoints/api/telemetry/instrumented\_inference\_runtime.py                 |       36 |        9 |        0 |        0 |     75% |     45-56 |
| src/emblema/entrypoints/api/telemetry/instrumented\_window\_tokeniser.py                  |       17 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/api/telemetry/telemetry.py                                        |       53 |        3 |        2 |        1 |     93% |     43-47 |
| src/emblema/entrypoints/cli/campaign/adapters.py                                          |        9 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/cli/campaign/campaign\_cli.py                                     |      128 |        1 |       20 |        1 |     99% |       250 |
| src/emblema/entrypoints/cli/campaign/campaign\_invocation.py                              |       13 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/cli/campaign/composition\_root.py                                 |       44 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/cli/campaign/known\_tasks.py                                      |       64 |       10 |       10 |        1 |     80% |66, 69-71, 129-133, 207 |
| src/emblema/entrypoints/cli/campaign/services.py                                          |       10 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/cli/campaign\_run/adapters.py                                     |        6 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/cli/campaign\_run/campaign\_run\_cli.py                           |       34 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/cli/campaign\_run/campaign\_run\_invocation.py                    |        4 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/cli/campaign\_run/composition\_root.py                            |       33 |        0 |        2 |        0 |    100% |           |
| src/emblema/entrypoints/cli/campaign\_run/services.py                                     |        4 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/cli/pretrain/adapters.py                                          |       11 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/cli/pretrain/composition\_root.py                                 |       55 |        0 |        4 |        0 |    100% |           |
| src/emblema/entrypoints/cli/pretrain/pretrain\_cli.py                                     |       91 |        0 |       18 |        0 |    100% |           |
| src/emblema/entrypoints/cli/pretrain/pretrain\_invocation.py                              |       12 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/cli/pretrain/services.py                                          |        7 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/cli/publish\_corpus/adapters.py                                   |        9 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/cli/publish\_corpus/composition\_root.py                          |       69 |        0 |       16 |        0 |    100% |           |
| src/emblema/entrypoints/cli/publish\_corpus/known\_corpora.py                             |       19 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/cli/publish\_corpus/publish\_corpus\_cli.py                       |       58 |        0 |       12 |        0 |    100% |           |
| src/emblema/entrypoints/cli/publish\_corpus/publish\_corpus\_invocation.py                |        6 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/cli/publish\_corpus/services.py                                   |        7 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/cli/serving/adapters.py                                           |        8 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/cli/serving/composition\_root.py                                  |       26 |        0 |        2 |        0 |    100% |           |
| src/emblema/entrypoints/cli/serving/services.py                                           |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/cli/serving/serving\_cli.py                                       |       42 |        0 |        6 |        1 |     98% | 74-\>exit |
| src/emblema/entrypoints/cli/serving/serving\_invocation.py                                |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/configured.py                                                     |       16 |        0 |        4 |        0 |    100% |           |
| src/emblema/entrypoints/known\_ground\_truths.py                                          |       14 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/restored\_backbones.py                                            |       19 |        0 |        2 |        0 |    100% |           |
| src/emblema/entrypoints/source\_revision.py                                               |       59 |        0 |       14 |        1 |     99% |   88-\>92 |
| src/emblema/entrypoints/workers/adapters.py                                               |       13 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/workers/campaign\_process.py                                      |       67 |        0 |        2 |        0 |    100% |           |
| src/emblema/entrypoints/workers/campaign\_worker.py                                       |       22 |        0 |        6 |        0 |    100% |           |
| src/emblema/entrypoints/workers/declared\_worker.py                                       |       32 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/workers/general/celery\_app.py                                    |       10 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/workers/general/composition\_root.py                              |       37 |        0 |        2 |        0 |    100% |           |
| src/emblema/entrypoints/workers/known\_arms.py                                            |       23 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/workers/known\_baselines.py                                       |       24 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/workers/known\_patch\_models.py                                   |       12 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/workers/ml/celery\_app.py                                         |       10 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/workers/ml/composition\_root.py                                   |       47 |        0 |        2 |        0 |    100% |           |
| src/emblema/entrypoints/workers/services.py                                               |        8 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/adapters/artifacts/kept\_candidates.py                             |       18 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/adapters/artifacts/kept\_classical\_inference.py                   |       52 |        0 |       10 |        1 |     98% |   79-\>81 |
| src/emblema/evaluation/adapters/artifacts/representation\_bytes.py                        |        4 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/adapters/blocks/block\_corpus\_windows.py                          |       24 |        0 |        2 |        0 |    100% |           |
| src/emblema/evaluation/adapters/blocks/published\_corpus\_blocks.py                       |       25 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/adapters/blocks/read\_corpus.py                                    |       25 |        0 |        4 |        0 |    100% |           |
| src/emblema/evaluation/adapters/campaigns/campaign\_file.py                               |       53 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/adapters/candidates/backbone\_arm.py                               |       21 |        0 |        4 |        0 |    100% |           |
| src/emblema/evaluation/adapters/candidates/backbone\_arm\_catalogue.py                    |       45 |        2 |       10 |        0 |     96% |     56-57 |
| src/emblema/evaluation/adapters/candidates/backbone\_candidate\_provider.py               |       20 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/adapters/candidates/classical\_arm.py                              |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/adapters/candidates/classical\_baseline\_catalogue.py              |       35 |        2 |        4 |        0 |     95% |     51-52 |
| src/emblema/evaluation/adapters/candidates/classical\_candidate\_provider.py              |       20 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/adapters/candidates/patch\_candidate\_provider.py                  |       18 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/adapters/candidates/patch\_model\_catalogue.py                     |       44 |        3 |        6 |        0 |     94% |101, 106-107 |
| src/emblema/evaluation/adapters/candidates/routed\_candidate\_catalogue.py                |       19 |        0 |        2 |        0 |    100% |           |
| src/emblema/evaluation/adapters/candidates/routed\_candidate\_provider.py                 |       23 |        2 |        2 |        0 |     92% |     48-49 |
| src/emblema/evaluation/adapters/documents/campaign\_candidate\_document.py                |       14 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/adapters/documents/campaign\_order\_json.py                        |       31 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/adapters/documents/campaign\_order\_result\_json.py                |       24 |        2 |        0 |        0 |     92% |     59-60 |
| src/emblema/evaluation/adapters/documents/cell\_result\_document.py                       |       21 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/adapters/documents/downstream\_task\_document.py                   |       30 |        4 |        6 |        1 |     75% |     81-84 |
| src/emblema/evaluation/adapters/documents/family\_correction\_document.py                 |       15 |        0 |        4 |        0 |    100% |           |
| src/emblema/evaluation/adapters/documents/stored\_document.py                             |       14 |        0 |        4 |        0 |    100% |           |
| src/emblema/evaluation/adapters/features/channel\_aggregated\_features.py                 |       36 |        0 |        2 |        0 |    100% |           |
| src/emblema/evaluation/adapters/features/per\_channel\_features.py                        |       29 |        0 |        6 |        0 |    100% |           |
| src/emblema/evaluation/adapters/features/spectral\_features.py                            |       29 |        0 |        8 |        1 |     97% |   57-\>64 |
| src/emblema/evaluation/adapters/features/window\_features.py                              |       13 |        0 |        6 |        1 |     95% | 29-\>exit |
| src/emblema/evaluation/adapters/features/window\_spectrum.py                              |       54 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/adapters/features/window\_statistics.py                            |       37 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/adapters/grid/regular\_grid.py                                     |       45 |        0 |        6 |        0 |    100% |           |
| src/emblema/evaluation/adapters/handoff/artifact\_store\_campaign\_handoff.py             |       24 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/adapters/in\_memory/adaptation\_runtime.py                         |       33 |        0 |        2 |        0 |    100% |           |
| src/emblema/evaluation/adapters/in\_memory/campaign\_handoff.py                           |       36 |        0 |        4 |        0 |    100% |           |
| src/emblema/evaluation/adapters/in\_memory/campaign\_listing.py                           |       30 |        0 |        4 |        0 |    100% |           |
| src/emblema/evaluation/adapters/in\_memory/candidate\_provider.py                         |       32 |        0 |        6 |        0 |    100% |           |
| src/emblema/evaluation/adapters/in\_memory/classical\_runtime.py                          |       40 |        0 |        4 |        0 |    100% |           |
| src/emblema/evaluation/adapters/in\_memory/corpus\_windows.py                             |       28 |        0 |        8 |        0 |    100% |           |
| src/emblema/evaluation/adapters/in\_memory/downstream\_task\_repository.py                |       13 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/adapters/in\_memory/evaluation\_campaign\_repository.py            |       18 |        0 |        2 |        0 |    100% |           |
| src/emblema/evaluation/adapters/in\_memory/ground\_truth.py                               |       20 |        0 |        8 |        0 |    100% |           |
| src/emblema/evaluation/adapters/in\_memory/patch\_runtime.py                              |       33 |        0 |        4 |        0 |    100% |           |
| src/emblema/evaluation/adapters/in\_memory/verdict\_memo.py                               |       25 |        0 |        6 |        0 |    100% |           |
| src/emblema/evaluation/adapters/minirocket/fitted\_convolutions.py                        |       55 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/adapters/minirocket/minirocket\_classical\_runtime.py              |       55 |        2 |       10 |        3 |     92% |70, 110-\>exit, 124 |
| src/emblema/evaluation/adapters/minirocket/minirocket\_transform.py                       |       72 |        1 |       16 |        1 |     98% |       111 |
| src/emblema/evaluation/adapters/onnx/inference\_candidate.py                              |       13 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/adapters/onnx/inference\_graph.py                                  |       92 |        1 |        4 |        0 |     99% |       152 |
| src/emblema/evaluation/adapters/persistence/campaign\_cell\_record.py                     |       34 |        0 |        2 |        0 |    100% |           |
| src/emblema/evaluation/adapters/persistence/campaign\_design\_document.py                 |       19 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/adapters/persistence/campaign\_listing.py                          |       32 |        0 |        6 |        0 |    100% |           |
| src/emblema/evaluation/adapters/persistence/campaign\_unit\_error\_record.py              |       24 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/adapters/persistence/downstream\_task\_record.py                   |       58 |        0 |       10 |        1 |     99% |152-\>exit |
| src/emblema/evaluation/adapters/persistence/downstream\_task\_repository.py               |       18 |        0 |        2 |        0 |    100% |           |
| src/emblema/evaluation/adapters/persistence/evaluation\_campaign\_record.py               |       37 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/adapters/persistence/evaluation\_campaign\_repository.py           |       24 |        0 |        6 |        0 |    100% |           |
| src/emblema/evaluation/adapters/persistence/orm.py                                        |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/adapters/persistence/task\_unit\_record.py                         |       19 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/adapters/readers/cmapss\_ground\_truth.py                          |       48 |        0 |       16 |        0 |    100% |           |
| src/emblema/evaluation/adapters/readers/corpus\_ground\_truths.py                         |       11 |        0 |        2 |        0 |    100% |           |
| src/emblema/evaluation/adapters/routing/method\_routed\_classical\_runtime.py             |       22 |        0 |        4 |        1 |     96% | 44-\>exit |
| src/emblema/evaluation/adapters/synthetic/known\_answer.py                                |       23 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/adapters/synthetic/synthetic\_ground\_truth.py                     |       33 |        0 |       10 |        0 |    100% |           |
| src/emblema/evaluation/adapters/torch/adapted\_backbone.py                                |       37 |        0 |       10 |        0 |    100% |           |
| src/emblema/evaluation/adapters/torch/attention\_pooling.py                               |       13 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/adapters/torch/backbone\_factory.py                                |        8 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/adapters/torch/fitted\_candidate.py                                |       27 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/adapters/torch/fitted\_patch\_model.py                             |       38 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/adapters/torch/grid\_reading.py                                    |       21 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/adapters/torch/lora\_linear.py                                     |       17 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/adapters/torch/low\_rank\_adaptation.py                            |       24 |        0 |        4 |        0 |    100% |           |
| src/emblema/evaluation/adapters/torch/mean\_pooling.py                                    |        9 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/adapters/torch/patch\_transformer.py                               |       35 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/adapters/torch/pooling.py                                          |       13 |        0 |        6 |        1 |     95% | 23-\>exit |
| src/emblema/evaluation/adapters/torch/regression\_head.py                                 |       10 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/adapters/torch/ridge\_solution.py                                  |       40 |        0 |        4 |        0 |    100% |           |
| src/emblema/evaluation/adapters/torch/scheduled\_training.py                              |       37 |        0 |        6 |        0 |    100% |           |
| src/emblema/evaluation/adapters/torch/tail\_pooling.py                                    |       11 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/adapters/torch/torch\_adaptation\_runtime.py                       |       77 |        0 |        6 |        0 |    100% |           |
| src/emblema/evaluation/adapters/torch/torch\_patch\_runtime.py                            |       56 |        0 |        4 |        0 |    100% |           |
| src/emblema/evaluation/adapters/xgboost/fitted\_baseline.py                               |       34 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/adapters/xgboost/xgboost\_classical\_runtime.py                    |       68 |        0 |       10 |        1 |     99% |111-\>exit |
| src/emblema/evaluation/api/campaign\_routes.py                                            |       39 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/api/evaluation\_refusals.py                                        |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/api/schemas/campaign\_detail.py                                    |       11 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/api/schemas/campaign\_resource.py                                  |        8 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/api/schemas/campaign\_run\_resource.py                             |        8 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/api/schemas/candidate\_curve\_resource.py                          |        9 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/api/schemas/comparison\_resource.py                                |        7 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/api/schemas/verdict\_resource.py                                   |        9 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/application/assemblers/campaign\_completed\_assembler.py           |       24 |        0 |        2 |        0 |    100% |           |
| src/emblema/evaluation/application/read\_models/campaign\_run.py                          |       10 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/application/read\_models/campaign\_summary.py                      |       13 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/application/read\_models/campaign\_view.py                         |        6 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/application/read\_models/candidate\_curve.py                       |       22 |        0 |        2 |        0 |    100% |           |
| src/emblema/evaluation/application/read\_models/comparison\_view.py                       |       10 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/application/read\_models/verdict\_view.py                          |       11 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/application/use\_cases/accept\_campaign\_order\_result.py          |       16 |        0 |        2 |        0 |    100% |           |
| src/emblema/evaluation/application/use\_cases/advance\_campaign.py                        |       30 |        0 |        4 |        0 |    100% |           |
| src/emblema/evaluation/application/use\_cases/announce\_campaign.py                       |       22 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/application/use\_cases/complete\_campaign.py                       |       29 |        0 |        2 |        0 |    100% |           |
| src/emblema/evaluation/application/use\_cases/define\_campaign.py                         |       48 |        0 |       10 |        0 |    100% |           |
| src/emblema/evaluation/application/use\_cases/define\_downstream\_task.py                 |       30 |        0 |        2 |        0 |    100% |           |
| src/emblema/evaluation/application/use\_cases/draw\_label\_budget.py                      |       25 |        1 |        2 |        1 |     93% |        64 |
| src/emblema/evaluation/application/use\_cases/draw\_run\_labels.py                        |       45 |        0 |        8 |        0 |    100% |           |
| src/emblema/evaluation/application/use\_cases/fulfil\_campaign\_order.py                  |       36 |        1 |        6 |        1 |     95% |        97 |
| src/emblema/evaluation/application/use\_cases/list\_campaign\_runs.py                     |       34 |        0 |        4 |        0 |    100% |           |
| src/emblema/evaluation/application/use\_cases/list\_campaigns.py                          |       34 |        0 |        2 |        0 |    100% |           |
| src/emblema/evaluation/application/use\_cases/open\_test\_split.py                        |       23 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/application/use\_cases/order\_campaign\_cells.py                   |       21 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/application/use\_cases/record\_cell\_result.py                     |       38 |        1 |       12 |        1 |     96% |        70 |
| src/emblema/evaluation/application/use\_cases/run\_adaptation.py                          |       21 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/application/use\_cases/run\_campaign\_cell.py                      |       24 |        0 |        4 |        0 |    100% |           |
| src/emblema/evaluation/application/use\_cases/run\_classical\_fit.py                      |       29 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/application/use\_cases/run\_patch\_training.py                     |       21 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/application/use\_cases/select\_tuned\_variants.py                  |       12 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/application/use\_cases/view\_campaign.py                           |       30 |        0 |        4 |        0 |    100% |           |
| src/emblema/evaluation/contracts/candidate\_kind.py                                       |       10 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/contracts/candidate\_metric.py                                     |       14 |        0 |        8 |        0 |    100% |           |
| src/emblema/evaluation/contracts/candidate\_standing.py                                   |        6 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/contracts/evaluated\_candidate.py                                  |        8 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/contracts/events.py                                                |        8 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/contracts/exceptions.py                                            |       11 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/contracts/identifiers.py                                           |       14 |        0 |        2 |        0 |    100% |           |
| src/emblema/evaluation/contracts/inference\_graph\_signature.py                           |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/contracts/kept\_candidate\_inference.py                            |        7 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/contracts/kept\_candidate\_manifest.py                             |       24 |        0 |       10 |        0 |    100% |           |
| src/emblema/evaluation/contracts/kept\_candidate\_manifest\_json.py                       |       69 |        3 |       16 |        3 |     93% |117, 133, 141 |
| src/emblema/evaluation/contracts/kept\_representation.py                                  |       11 |        0 |        4 |        0 |    100% |           |
| src/emblema/evaluation/domain/campaign/campaign\_candidate.py                             |       16 |        0 |        4 |        0 |    100% |           |
| src/emblema/evaluation/domain/campaign/campaign\_cell.py                                  |        7 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/domain/campaign/campaign\_design.py                                |       89 |        2 |       46 |        2 |     97% |    83, 86 |
| src/emblema/evaluation/domain/campaign/campaign\_overview.py                              |       27 |        0 |        4 |        0 |    100% |           |
| src/emblema/evaluation/domain/campaign/campaign\_position.py                              |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/domain/campaign/campaign\_verdict.py                               |       68 |        0 |       26 |        0 |    100% |           |
| src/emblema/evaluation/domain/campaign/candidate\_comparison.py                           |        9 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/domain/campaign/candidate\_evaluation.py                           |       15 |        0 |        4 |        0 |    100% |           |
| src/emblema/evaluation/domain/campaign/candidate\_method.py                               |       40 |        0 |       12 |        0 |    100% |           |
| src/emblema/evaluation/domain/campaign/cell\_result.py                                    |       26 |        0 |        6 |        0 |    100% |           |
| src/emblema/evaluation/domain/campaign/compute\_budget.py                                 |       17 |        0 |        6 |        0 |    100% |           |
| src/emblema/evaluation/domain/campaign/evaluation\_campaign.py                            |      133 |        1 |       40 |        1 |     99% |       217 |
| src/emblema/evaluation/domain/classical/boosted\_trees.py                                 |       21 |        0 |        2 |        0 |    100% |           |
| src/emblema/evaluation/domain/classical/classical\_method.py                              |        3 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/domain/classical/classical\_recipe.py                              |       14 |        0 |        4 |        0 |    100% |           |
| src/emblema/evaluation/domain/classical/feature\_scheme.py                                |        8 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/domain/classical/fitting\_source.py                                |        7 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/domain/classical/gradient\_boosting\_spec.py                       |       19 |        0 |       14 |        0 |    100% |           |
| src/emblema/evaluation/domain/classical/minirocket\_spec.py                               |       18 |        0 |        4 |        0 |    100% |           |
| src/emblema/evaluation/domain/classical/random\_convolutions.py                           |       22 |        0 |        2 |        0 |    100% |           |
| src/emblema/evaluation/domain/classical/ridge\_spec.py                                    |       14 |        0 |        2 |        0 |    100% |           |
| src/emblema/evaluation/domain/exceptions.py                                               |       90 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/domain/handoff/campaign\_order.py                                  |       36 |        0 |       16 |        0 |    100% |           |
| src/emblema/evaluation/domain/handoff/campaign\_order\_result.py                          |       22 |        0 |        6 |        0 |    100% |           |
| src/emblema/evaluation/domain/heads/head\_pooling.py                                      |       31 |        0 |        6 |        0 |    100% |           |
| src/emblema/evaluation/domain/heads/ridge\_penalties.py                                   |       15 |        0 |        8 |        0 |    100% |           |
| src/emblema/evaluation/domain/identifiers.py                                              |        9 |        0 |        2 |        0 |    100% |           |
| src/emblema/evaluation/domain/labels/forecast\_scheme.py                                  |       17 |        0 |        4 |        0 |    100% |           |
| src/emblema/evaluation/domain/labels/label\_budget.py                                     |       32 |        0 |        8 |        0 |    100% |           |
| src/emblema/evaluation/domain/labels/label\_sample.py                                     |       41 |        0 |       12 |        1 |     98% |   76-\>75 |
| src/emblema/evaluation/domain/labels/labelled\_window.py                                  |        4 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/domain/labels/remaining\_life\_scheme.py                           |       16 |        0 |        4 |        0 |    100% |           |
| src/emblema/evaluation/domain/labels/run\_labels.py                                       |        6 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/domain/labels/target\_bins.py                                      |       21 |        0 |        6 |        0 |    100% |           |
| src/emblema/evaluation/domain/labels/task\_window.py                                      |       11 |        0 |        4 |        0 |    100% |           |
| src/emblema/evaluation/domain/patching/patch\_model\_spec.py                              |       32 |        1 |       16 |        1 |     96% |        92 |
| src/emblema/evaluation/domain/patching/patch\_plan.py                                     |        9 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/domain/scoring/scored\_outcome.py                                  |       21 |        0 |        6 |        0 |    100% |           |
| src/emblema/evaluation/domain/scoring/unit\_error.py                                      |       32 |        0 |       10 |        0 |    100% |           |
| src/emblema/evaluation/domain/scoring/window\_prediction.py                               |       13 |        0 |        4 |        0 |    100% |           |
| src/emblema/evaluation/domain/statistics/benjamini\_hochberg\_correction.py               |       29 |        0 |       16 |        0 |    100% |           |
| src/emblema/evaluation/domain/statistics/bootstrap\_interval.py                           |       20 |        0 |        4 |        0 |    100% |           |
| src/emblema/evaluation/domain/statistics/comparison\_rules.py                             |       39 |        0 |       16 |        0 |    100% |           |
| src/emblema/evaluation/domain/statistics/comparison\_verdict.py                           |       10 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/domain/statistics/error\_over\_repeats.py                          |       28 |        0 |       16 |        0 |    100% |           |
| src/emblema/evaluation/domain/statistics/family\_correction.py                            |        3 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/domain/statistics/holm\_correction.py                              |       27 |        0 |       14 |        0 |    100% |           |
| src/emblema/evaluation/domain/statistics/paired\_difference.py                            |       17 |        0 |        6 |        0 |    100% |           |
| src/emblema/evaluation/domain/statistics/paired\_unit\_bootstrap.py                       |       35 |        0 |        4 |        0 |    100% |           |
| src/emblema/evaluation/domain/statistics/paired\_unit\_errors.py                          |       66 |        0 |       22 |        0 |    100% |           |
| src/emblema/evaluation/domain/statistics/practical\_floor.py                              |       17 |        0 |        4 |        0 |    100% |           |
| src/emblema/evaluation/domain/statistics/two\_level\_bootstrap.py                         |       46 |        0 |        8 |        0 |    100% |           |
| src/emblema/evaluation/domain/statistics/uncertainty\_decomposition.py                    |       14 |        0 |        4 |        0 |    100% |           |
| src/emblema/evaluation/domain/task/corpus\_sides.py                                       |       13 |        0 |        6 |        0 |    100% |           |
| src/emblema/evaluation/domain/task/downstream\_task.py                                    |       60 |        0 |       20 |        1 |     99% | 123-\>125 |
| src/emblema/evaluation/domain/task/evaluation\_protocol.py                                |        7 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/domain/task/frozen\_test\_split.py                                 |       10 |        0 |        4 |        0 |    100% |           |
| src/emblema/evaluation/domain/task/inner\_holdout.py                                      |       18 |        0 |        4 |        0 |    100% |           |
| src/emblema/evaluation/domain/task/run\_purpose.py                                        |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/domain/task/task\_split.py                                         |       15 |        0 |        8 |        0 |    100% |           |
| src/emblema/evaluation/domain/transfer/adaptation\_outcome.py                             |       26 |        4 |       12 |        4 |     79% |75, 79, 85, 89 |
| src/emblema/evaluation/domain/transfer/adaptation\_plan.py                                |       23 |        0 |        8 |        0 |    100% |           |
| src/emblema/evaluation/domain/transfer/adaptation\_schedule.py                            |       37 |        0 |       18 |        0 |    100% |           |
| src/emblema/evaluation/domain/transfer/lora\_spec.py                                      |       22 |        0 |       14 |        0 |    100% |           |
| src/emblema/evaluation/domain/transfer/remaining\_life\_metrics.py                        |       43 |        0 |       18 |        0 |    100% |           |
| src/emblema/evaluation/domain/transfer/transfer\_mode.py                                  |       19 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/domain/tuning/candidate\_variant.py                                |       49 |        1 |       26 |        1 |     97% |        35 |
| src/emblema/evaluation/domain/tuning/knob.py                                              |       13 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/domain/tuning/one\_standard\_error\_rule.py                        |       20 |        1 |        4 |        1 |     92% |        55 |
| src/emblema/evaluation/domain/tuning/tuned\_choice.py                                     |       10 |        0 |        2 |        0 |    100% |           |
| src/emblema/evaluation/ports/adaptation\_runtime.py                                       |        9 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/ports/campaign\_handoff.py                                         |        9 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/ports/campaign\_listing.py                                         |        9 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/ports/candidate\_catalogue.py                                      |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/ports/candidate\_provider.py                                       |        8 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/ports/classical\_runtime.py                                        |       10 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/ports/corpus\_windows.py                                           |        9 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/ports/downstream\_task\_repository.py                              |        6 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/ports/evaluation\_campaign\_repository.py                          |        6 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/ports/ground\_truth.py                                             |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/ports/patch\_runtime.py                                            |        9 |        0 |        0 |        0 |    100% |           |
| src/emblema/evaluation/ports/verdict\_memo.py                                             |        6 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/adapters/blocks/block\_training\_corpus\_reader.py                |       39 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/adapters/diagnostics/channel\_series.py                           |       24 |        0 |        2 |        0 |    100% |           |
| src/emblema/pretraining/adapters/diagnostics/cross\_channel\_ridge\_baseline.py           |       47 |        0 |       10 |        0 |    100% |           |
| src/emblema/pretraining/adapters/diagnostics/linear\_interpolation\_baseline.py           |       19 |        0 |        6 |        0 |    100% |           |
| src/emblema/pretraining/adapters/diagnostics/own\_and\_cross\_channel\_ridge\_baseline.py |       63 |        0 |       16 |        1 |     99% | 129-\>127 |
| src/emblema/pretraining/adapters/diagnostics/ridge\_regression.py                         |       24 |        0 |       10 |        0 |    100% |           |
| src/emblema/pretraining/adapters/diagnostics/spectral\_recovery.py                        |       57 |        0 |       10 |        0 |    100% |           |
| src/emblema/pretraining/adapters/diagnostics/triviality\_diagnostic.py                    |       61 |        0 |       16 |        0 |    100% |           |
| src/emblema/pretraining/adapters/diagnostics/unit\_bootstrap.py                           |       39 |        0 |        8 |        0 |    100% |           |
| src/emblema/pretraining/adapters/diagnostics/window\_arrays.py                            |       19 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/adapters/documents/experiment\_configuration\_document.py         |       19 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/adapters/documents/fields.py                                      |       69 |        0 |       14 |        0 |    100% |           |
| src/emblema/pretraining/adapters/documents/pretraining\_order\_json.py                    |       30 |        0 |        4 |        0 |    100% |           |
| src/emblema/pretraining/adapters/documents/pretraining\_result\_json.py                   |       47 |        0 |        4 |        0 |    100% |           |
| src/emblema/pretraining/adapters/encoder/encoder\_block.py                                |       15 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/adapters/encoder/fourier\_time\_encoding.py                       |       15 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/adapters/encoder/grown\_channel\_embedding.py                     |       24 |        0 |        2 |        0 |    100% |           |
| src/emblema/pretraining/adapters/encoder/learned\_channel\_embedding.py                   |       14 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/adapters/encoder/self\_attention.py                               |       21 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/adapters/encoder/set\_encoder.py                                  |       34 |        0 |        6 |        0 |    100% |           |
| src/emblema/pretraining/adapters/encoder/tier\_architecture.py                            |        4 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/adapters/experiments/experiment\_file.py                          |       51 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/adapters/handoff/artifact\_store\_handoff\_exchange.py            |       20 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/adapters/handoff/handoff\_training\_runtime.py                    |       19 |        0 |        2 |        0 |    100% |           |
| src/emblema/pretraining/adapters/in\_memory/backbone\_repository.py                       |       13 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/adapters/in\_memory/experiment\_tracker.py                        |       27 |        0 |        6 |        0 |    100% |           |
| src/emblema/pretraining/adapters/in\_memory/handoff\_exchange.py                          |       33 |        0 |        4 |        0 |    100% |           |
| src/emblema/pretraining/adapters/in\_memory/training\_corpus\_reader.py                   |       34 |        0 |        6 |        0 |    100% |           |
| src/emblema/pretraining/adapters/in\_memory/training\_runtime.py                          |       57 |        0 |       12 |        0 |    100% |           |
| src/emblema/pretraining/adapters/mlflow/mlflow\_experiment\_tracker.py                    |       40 |        0 |       10 |        0 |    100% |           |
| src/emblema/pretraining/adapters/objective/masked\_reconstruction.py                      |       15 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/adapters/objective/reconstruction\_decoder.py                     |       20 |        0 |        4 |        0 |    100% |           |
| src/emblema/pretraining/adapters/objective/reconstruction\_loss.py                        |       30 |        0 |        2 |        0 |    100% |           |
| src/emblema/pretraining/adapters/objective/token\_masking.py                              |       35 |        0 |        2 |        0 |    100% |           |
| src/emblema/pretraining/adapters/objective/token\_masks.py                                |       18 |        0 |        4 |        0 |    100% |           |
| src/emblema/pretraining/adapters/persistence/backbone\_record.py                          |       50 |        0 |        2 |        0 |    100% |           |
| src/emblema/pretraining/adapters/persistence/backbone\_repository.py                      |       18 |        0 |        2 |        0 |    100% |           |
| src/emblema/pretraining/adapters/persistence/orm.py                                       |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/adapters/persistence/pretraining\_input\_record.py                |       29 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/adapters/training/device\_generator.py                            |       14 |        3 |        4 |        1 |     67% | 22, 31-32 |
| src/emblema/pretraining/adapters/training/devices.py                                      |        5 |        0 |        2 |        0 |    100% |           |
| src/emblema/pretraining/adapters/training/exceptions.py                                   |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/adapters/training/torch\_precision.py                             |       26 |        0 |        4 |        0 |    100% |           |
| src/emblema/pretraining/adapters/training/torch\_training\_runtime.py                     |      216 |        0 |       50 |        2 |     99% |389-\>391, 437-\>436 |
| src/emblema/pretraining/adapters/training/trained\_model.py                               |       33 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/adapters/training/training\_checkpoint.py                         |       40 |        0 |       10 |        0 |    100% |           |
| src/emblema/pretraining/application/use\_cases/accept\_pretraining\_result.py             |       30 |        0 |        2 |        0 |    100% |           |
| src/emblema/pretraining/application/use\_cases/assess\_reconstruction\_run.py             |       11 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/application/use\_cases/fulfil\_pretraining\_order.py              |       22 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/application/use\_cases/order\_pretraining.py                      |       34 |        0 |        4 |        0 |    100% |           |
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
| src/emblema/pretraining/domain/backbone/backbone.py                                       |       53 |        0 |       16 |        0 |    100% |           |
| src/emblema/pretraining/domain/backbone/pretraining\_input.py                             |       12 |        0 |        4 |        0 |    100% |           |
| src/emblema/pretraining/domain/encoder\_architecture.py                                   |       29 |        0 |        8 |        0 |    100% |           |
| src/emblema/pretraining/domain/exceptions.py                                              |       54 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/domain/handoff/pretraining\_order.py                              |       29 |        0 |       12 |        0 |    100% |           |
| src/emblema/pretraining/domain/handoff/pretraining\_result.py                             |       51 |        0 |       16 |        0 |    100% |           |
| src/emblema/pretraining/domain/identifiers.py                                             |        4 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/domain/mask\_kind.py                                              |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/domain/masking\_strategy.py                                       |       18 |        0 |       10 |        0 |    100% |           |
| src/emblema/pretraining/domain/saturation/saturation\_curve.py                            |       22 |        0 |        8 |        0 |    100% |           |
| src/emblema/pretraining/domain/saturation/saturation\_point.py                            |       27 |        0 |       14 |        0 |    100% |           |
| src/emblema/pretraining/domain/saturation/saturation\_verdict.py                          |       25 |        0 |        6 |        0 |    100% |           |
| src/emblema/pretraining/domain/training/checkpoint\_policy.py                             |       11 |        0 |        4 |        0 |    100% |           |
| src/emblema/pretraining/domain/training/corpus\_share.py                                  |       22 |        0 |        6 |        0 |    100% |           |
| src/emblema/pretraining/domain/training/corpus\_validation.py                             |       17 |        0 |        8 |        0 |    100% |           |
| src/emblema/pretraining/domain/training/epoch\_outcome.py                                 |       30 |        0 |       12 |        0 |    100% |           |
| src/emblema/pretraining/domain/training/experiment\_configuration.py                      |       32 |        0 |        6 |        0 |    100% |           |
| src/emblema/pretraining/domain/training/objective\_loss.py                                |       26 |        0 |       10 |        0 |    100% |           |
| src/emblema/pretraining/domain/training/precision.py                                      |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/domain/training/run\_position.py                                  |       16 |        0 |        4 |        0 |    100% |           |
| src/emblema/pretraining/domain/training/run\_signature.py                                 |       26 |        0 |        2 |        0 |    100% |           |
| src/emblema/pretraining/domain/training/training\_budget.py                               |       32 |        0 |       16 |        0 |    100% |           |
| src/emblema/pretraining/domain/training/training\_corpus.py                               |       24 |        0 |        6 |        0 |    100% |           |
| src/emblema/pretraining/domain/training/training\_corpus\_shape.py                        |       13 |        0 |        8 |        0 |    100% |           |
| src/emblema/pretraining/domain/training/training\_mixture.py                              |       28 |        0 |        4 |        0 |    100% |           |
| src/emblema/pretraining/domain/training/training\_mixture\_shape.py                       |       21 |        0 |        6 |        0 |    100% |           |
| src/emblema/pretraining/domain/training/training\_outcome.py                              |       24 |        0 |        8 |        0 |    100% |           |
| src/emblema/pretraining/ports/backbone\_repository.py                                     |        6 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/ports/experiment\_tracker.py                                      |        8 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/ports/handoff\_exchange.py                                        |        9 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/ports/mask\_kind\_summariser.py                                   |        6 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/ports/training\_corpus\_reader.py                                 |        8 |        0 |        0 |        0 |    100% |           |
| src/emblema/pretraining/ports/training\_runtime.py                                        |        8 |        0 |        0 |        0 |    100% |           |
| src/emblema/serving/adapters/acl/evaluation.py                                            |       18 |        0 |        0 |        0 |    100% |           |
| src/emblema/serving/adapters/in\_memory/inference\_runtime.py                             |       31 |        0 |        2 |        0 |    100% |           |
| src/emblema/serving/adapters/in\_memory/promotable\_artifact\_repository.py               |       10 |        0 |        0 |        0 |    100% |           |
| src/emblema/serving/adapters/in\_memory/served\_model\_listing.py                         |       21 |        0 |        4 |        0 |    100% |           |
| src/emblema/serving/adapters/in\_memory/served\_model\_repository.py                      |       18 |        0 |        2 |        0 |    100% |           |
| src/emblema/serving/adapters/onnx/onnx\_graph\_inference.py                               |       77 |        0 |       16 |        0 |    100% |           |
| src/emblema/serving/adapters/onnx/weighted\_semaphore.py                                  |       53 |        0 |       10 |        0 |    100% |           |
| src/emblema/serving/adapters/persistence/campaign\_score\_record.py                       |       22 |        0 |        0 |        0 |    100% |           |
| src/emblema/serving/adapters/persistence/orm.py                                           |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/serving/adapters/persistence/promotable\_artifact\_record.py                  |       34 |        0 |        0 |        0 |    100% |           |
| src/emblema/serving/adapters/persistence/promotable\_artifact\_repository.py              |       15 |        0 |        0 |        0 |    100% |           |
| src/emblema/serving/adapters/persistence/served\_model\_listing.py                        |       25 |        0 |        6 |        0 |    100% |           |
| src/emblema/serving/adapters/persistence/served\_model\_record.py                         |       33 |        0 |        0 |        0 |    100% |           |
| src/emblema/serving/adapters/persistence/served\_model\_repository.py                     |       28 |        1 |        4 |        1 |     94% |        41 |
| src/emblema/serving/adapters/routing/format\_routed\_inference\_runtime.py                |       81 |        0 |       16 |        0 |    100% |           |
| src/emblema/serving/api/schemas/embeddings\_resource.py                                   |        8 |        0 |        0 |        0 |    100% |           |
| src/emblema/serving/api/schemas/inference\_request.py                                     |        4 |        0 |        0 |        0 |    100% |           |
| src/emblema/serving/api/schemas/model\_input\_resource.py                                 |        9 |        0 |        0 |        0 |    100% |           |
| src/emblema/serving/api/schemas/observed\_window\_body.py                                 |       18 |        0 |        0 |        0 |    100% |           |
| src/emblema/serving/api/schemas/predictions\_resource.py                                  |       10 |        0 |        0 |        0 |    100% |           |
| src/emblema/serving/api/schemas/served\_model\_detail.py                                  |        9 |        0 |        0 |        0 |    100% |           |
| src/emblema/serving/api/schemas/served\_model\_resource.py                                |       11 |        0 |        0 |        0 |    100% |           |
| src/emblema/serving/api/served\_model\_routes.py                                          |       49 |        0 |        0 |        0 |    100% |           |
| src/emblema/serving/api/serving\_refusals.py                                              |        6 |        0 |        0 |        0 |    100% |           |
| src/emblema/serving/application/admission/prepared\_window.py                             |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/serving/application/admission/window\_admission.py                            |       38 |        0 |        4 |        0 |    100% |           |
| src/emblema/serving/application/read\_models/embedded\_window.py                          |        8 |        0 |        2 |        0 |    100% |           |
| src/emblema/serving/application/read\_models/embeddings.py                                |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/serving/application/read\_models/model\_input\_view.py                        |       10 |        0 |        0 |        0 |    100% |           |
| src/emblema/serving/application/read\_models/predicted\_window.py                         |        8 |        0 |        2 |        0 |    100% |           |
| src/emblema/serving/application/read\_models/predictions.py                               |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/serving/application/read\_models/served\_model\_summary.py                    |       14 |        0 |        0 |        0 |    100% |           |
| src/emblema/serving/application/read\_models/served\_model\_view.py                       |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/serving/application/use\_cases/embed\_windows.py                              |       21 |        0 |        0 |        0 |    100% |           |
| src/emblema/serving/application/use\_cases/list\_served\_models.py                        |       36 |        0 |        2 |        0 |    100% |           |
| src/emblema/serving/application/use\_cases/predict\_windows.py                            |       21 |        0 |        0 |        0 |    100% |           |
| src/emblema/serving/application/use\_cases/promote\_artifact.py                           |       49 |        0 |       10 |        0 |    100% |           |
| src/emblema/serving/application/use\_cases/record\_promotable\_artifacts.py               |       11 |        0 |        2 |        0 |    100% |           |
| src/emblema/serving/application/use\_cases/view\_served\_model.py                         |       16 |        0 |        0 |        0 |    100% |           |
| src/emblema/serving/application/use\_cases/withdraw\_served\_model.py                     |       13 |        0 |        0 |        0 |    100% |           |
| src/emblema/serving/domain/admitted\_window.py                                            |       14 |        0 |        4 |        0 |    100% |           |
| src/emblema/serving/domain/artifact\_origin.py                                            |        4 |        0 |        0 |        0 |    100% |           |
| src/emblema/serving/domain/campaign\_score.py                                             |       14 |        0 |        8 |        0 |    100% |           |
| src/emblema/serving/domain/exceptions.py                                                  |       26 |        0 |        0 |        0 |    100% |           |
| src/emblema/serving/domain/identifiers.py                                                 |        4 |        0 |        0 |        0 |    100% |           |
| src/emblema/serving/domain/inference\_budget.py                                           |       28 |        0 |       10 |        0 |    100% |           |
| src/emblema/serving/domain/inference\_limits.py                                           |       16 |        0 |       10 |        0 |    100% |           |
| src/emblema/serving/domain/model\_input.py                                                |       36 |        0 |       10 |        0 |    100% |           |
| src/emblema/serving/domain/promotable\_artifact.py                                        |       14 |        0 |        2 |        0 |    100% |           |
| src/emblema/serving/domain/served\_model.py                                               |       33 |        0 |       10 |        0 |    100% |           |
| src/emblema/serving/domain/served\_model\_position.py                                     |       10 |        0 |        0 |        0 |    100% |           |
| src/emblema/serving/domain/served\_model\_state.py                                        |        4 |        0 |        0 |        0 |    100% |           |
| src/emblema/serving/ports/inference\_runtime.py                                           |        9 |        0 |        0 |        0 |    100% |           |
| src/emblema/serving/ports/promotable\_artifact\_repository.py                             |        6 |        0 |        0 |        0 |    100% |           |
| src/emblema/serving/ports/served\_model\_listing.py                                       |        6 |        0 |        0 |        0 |    100% |           |
| src/emblema/serving/ports/served\_model\_repository.py                                    |        6 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/arrays/token\_batch.py                                        |       38 |        0 |        4 |        0 |    100% |           |
| src/emblema/shared/adapters/in\_memory/artifact\_store.py                                 |       28 |        0 |        6 |        0 |    100% |           |
| src/emblema/shared/adapters/in\_memory/clock.py                                           |        6 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/in\_memory/event\_publisher.py                                |        8 |        0 |        2 |        0 |    100% |           |
| src/emblema/shared/adapters/in\_memory/event\_subscriber.py                               |       10 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/in\_memory/id\_generator.py                                   |        9 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/loaders/interleaved\_loader.py                                |       27 |        0 |        6 |        0 |    100% |           |
| src/emblema/shared/adapters/loaders/seeded\_shuffle\_sampler.py                           |       18 |        0 |        2 |        0 |    100% |           |
| src/emblema/shared/adapters/loaders/window\_dataset.py                                    |       12 |        0 |        2 |        0 |    100% |           |
| src/emblema/shared/adapters/loaders/window\_loader.py                                     |       20 |        0 |        2 |        0 |    100% |           |
| src/emblema/shared/adapters/persistence/datetimes.py                                      |        7 |        0 |        2 |        0 |    100% |           |
| src/emblema/shared/adapters/persistence/naming.py                                         |        2 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/queues/celery\_application.py                                 |       10 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/queues/celery\_job\_queue.py                                  |       12 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/queues/immediate\_job\_queue.py                               |       14 |        0 |        2 |        0 |    100% |           |
| src/emblema/shared/adapters/storage/files.py                                              |       29 |        0 |        6 |        0 |    100% |           |
| src/emblema/shared/adapters/storage/layout.py                                             |       11 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/storage/local\_directory.py                                   |       55 |        0 |       10 |        0 |    100% |           |
| src/emblema/shared/adapters/storage/s3.py                                                 |       65 |        0 |       14 |        1 |     99% |   80-\>85 |
| src/emblema/shared/adapters/synthetic/dials.py                                            |        6 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/synthetic/draws.py                                            |       16 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/synthetic/latent\_factor\_process.py                          |       26 |        0 |        2 |        0 |    100% |           |
| src/emblema/shared/adapters/synthetic/layouts.py                                          |       13 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/synthetic/sensor\_layout.py                                   |       29 |        0 |        4 |        0 |    100% |           |
| src/emblema/shared/adapters/synthetic/sensor\_signal.py                                   |       35 |        0 |        2 |        0 |    100% |           |
| src/emblema/shared/adapters/system/clock.py                                               |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/system/id\_generator.py                                       |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/tensors/grown\_parameters.py                                  |        6 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/tensors/masked\_mean\_pooling.py                              |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/tensors/token\_tensors.py                                     |       31 |        0 |        2 |        0 |    100% |           |
| src/emblema/shared/adapters/windows/block\_workspace.py                                   |       22 |        0 |        4 |        0 |    100% |           |
| src/emblema/shared/adapters/windows/exceptions.py                                         |        9 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/windows/format.py                                             |       29 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/windows/window\_block.py                                      |       79 |        0 |       12 |        0 |    100% |           |
| src/emblema/shared/adapters/windows/window\_block\_writer.py                              |      118 |        0 |       26 |        1 |     99% | 251-\>255 |
| src/emblema/shared/api/cursor\_page.py                                                    |       10 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/api/cursor\_token.py                                                   |       25 |        0 |        4 |        0 |    100% |           |
| src/emblema/shared/api/page\_request.py                                                   |       18 |        0 |        2 |        0 |    100% |           |
| src/emblema/shared/api/problem.py                                                         |       12 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/events/domain\_event.py                                                |        8 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/jobs/job\_argument.py                                                  |        1 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/jobs/queued\_job.py                                                    |        6 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/jobs/worker\_pool.py                                                   |        4 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/kernel/artifacts.py                                                    |        8 |        0 |        2 |        0 |    100% |           |
| src/emblema/shared/kernel/checksums.py                                                    |       42 |        0 |        8 |        0 |    100% |           |
| src/emblema/shared/kernel/compute.py                                                      |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/kernel/exceptions.py                                                   |       16 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/kernel/identifiers.py                                                  |       17 |        0 |        2 |        0 |    100% |           |
| src/emblema/shared/kernel/learning\_rate\_schedule.py                                     |       21 |        0 |       10 |        0 |    100% |           |
| src/emblema/shared/kernel/ordering.py                                                     |       10 |        0 |        2 |        0 |    100% |           |
| src/emblema/shared/kernel/paging/cursor.py                                                |       11 |        0 |        4 |        0 |    100% |           |
| src/emblema/shared/kernel/paging/page.py                                                  |        8 |        0 |        2 |        0 |    100% |           |
| src/emblema/shared/kernel/retention.py                                                    |        4 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/kernel/sampling.py                                                     |        4 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/kernel/timestamps.py                                                   |       11 |        0 |        4 |        0 |    100% |           |
| src/emblema/shared/kernel/tokens.py                                                       |       47 |        0 |       20 |        0 |    100% |           |
| src/emblema/shared/ports/artifact\_store.py                                               |        9 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/ports/clock.py                                                         |        3 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/ports/event\_publisher.py                                              |        3 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/ports/event\_subscriber.py                                             |        4 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/ports/exceptions.py                                                    |        4 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/ports/id\_generator.py                                                 |        3 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/ports/job\_queue.py                                                    |        4 |        0 |        0 |        0 |    100% |           |
| **TOTAL**                                                                                 | **14185** |   **64** | **2442** |   **48** | **99%** |           |

127 empty files skipped.


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