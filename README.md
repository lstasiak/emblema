# Repository Coverage



| Name                                                                                 |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/emblema/catalog/adapters/archive/block\_corpus\_archive.py                       |       76 |        0 |        8 |        0 |    100% |           |
| src/emblema/catalog/adapters/in\_memory/corpus\_reader.py                            |       29 |        0 |        6 |        0 |    100% |           |
| src/emblema/catalog/adapters/in\_memory/corpus\_repository.py                        |       18 |        0 |        4 |        0 |    100% |           |
| src/emblema/catalog/adapters/persistence/corpus\_record.py                           |       21 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/adapters/persistence/corpus\_repository.py                       |       33 |        1 |        4 |        1 |     95% |        46 |
| src/emblema/catalog/adapters/persistence/corpus\_version\_record.py                  |       45 |        0 |        2 |        0 |    100% |           |
| src/emblema/catalog/adapters/persistence/orm.py                                      |        4 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/adapters/readers/cmapss.py                                       |      110 |        0 |       46 |        0 |    100% |           |
| src/emblema/catalog/adapters/tokenisation/sliding\_window.py                         |      114 |        0 |       32 |        0 |    100% |           |
| src/emblema/catalog/application/assemblers/corpus\_version\_ref\_assembler.py        |        6 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/application/assemblers/published\_corpus\_manifest\_assembler.py |       28 |        0 |        2 |        0 |    100% |           |
| src/emblema/catalog/application/use\_cases/publish\_corpus.py                        |       39 |        0 |        4 |        0 |    100% |           |
| src/emblema/catalog/application/use\_cases/register\_corpus.py                       |       16 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/application/use\_cases/register\_corpus\_version.py              |       33 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/application/use\_cases/tokenise\_corpus\_version.py              |       52 |        1 |        4 |        1 |     96% |        87 |
| src/emblema/catalog/contracts/corpus\_version\_ref.py                                |       24 |        0 |        8 |        0 |    100% |           |
| src/emblema/catalog/contracts/events.py                                              |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/contracts/exceptions.py                                          |        7 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/contracts/identifiers.py                                         |        4 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/contracts/published\_channel.py                                  |        7 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/contracts/published\_channel\_statistics.py                      |        3 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/contracts/published\_corpus\_manifest.py                         |       29 |        0 |       16 |        0 |    100% |           |
| src/emblema/catalog/contracts/published\_corpus\_manifest\_json.py                   |       98 |        1 |       22 |        1 |     98% |       164 |
| src/emblema/catalog/domain/archived\_corpus.py                                       |       15 |        0 |        8 |        0 |    100% |           |
| src/emblema/catalog/domain/channel\_schema.py                                        |       27 |        0 |        6 |        0 |    100% |           |
| src/emblema/catalog/domain/channel\_statistics.py                                    |       19 |        0 |        6 |        0 |    100% |           |
| src/emblema/catalog/domain/channel\_vocabulary.py                                    |       48 |        0 |       24 |        0 |    100% |           |
| src/emblema/catalog/domain/corpus.py                                                 |       54 |        0 |       20 |        0 |    100% |           |
| src/emblema/catalog/domain/corpus\_content.py                                        |       10 |        0 |        4 |        0 |    100% |           |
| src/emblema/catalog/domain/corpus\_description.py                                    |        6 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/domain/corpus\_source.py                                         |        9 |        0 |        4 |        0 |    100% |           |
| src/emblema/catalog/domain/corpus\_unit.py                                           |       26 |        0 |        6 |        0 |    100% |           |
| src/emblema/catalog/domain/corpus\_version.py                                        |       51 |        0 |       14 |        0 |    100% |           |
| src/emblema/catalog/domain/exceptions.py                                             |       84 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/domain/identifiers.py                                            |       12 |        0 |        2 |        0 |    100% |           |
| src/emblema/catalog/domain/licence.py                                                |       10 |        0 |        4 |        0 |    100% |           |
| src/emblema/catalog/domain/observation.py                                            |       12 |        0 |        6 |        0 |    100% |           |
| src/emblema/catalog/domain/placed\_window.py                                         |        6 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/domain/static\_feature.py                                        |       10 |        0 |        4 |        0 |    100% |           |
| src/emblema/catalog/domain/tokenisation\_manifest.py                                 |       36 |        0 |        8 |        0 |    100% |           |
| src/emblema/catalog/domain/tokenisation\_scheme.py                                   |       46 |        0 |       10 |        0 |    100% |           |
| src/emblema/catalog/domain/unit\_split.py                                            |       26 |        0 |       10 |        0 |    100% |           |
| src/emblema/catalog/domain/window\_reconstruction.py                                 |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/domain/window\_spec.py                                           |       21 |        0 |        6 |        0 |    100% |           |
| src/emblema/catalog/ports/corpus\_archive.py                                         |       13 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/ports/corpus\_reader.py                                          |       10 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/ports/corpus\_repository.py                                      |        7 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/ports/tokeniser.py                                               |       11 |        0 |        0 |        0 |    100% |           |
| src/emblema/config/artifact\_store\_settings.py                                      |        8 |        0 |        0 |        0 |    100% |           |
| src/emblema/config/database\_settings.py                                             |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/config/settings.py                                                       |       13 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/cli/adapters.py                                              |        9 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/cli/composition\_root.py                                     |       43 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/cli/known\_corpora.py                                        |       16 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/cli/publish\_corpus.py                                       |       37 |        0 |        2 |        0 |    100% |           |
| src/emblema/entrypoints/cli/publish\_corpus\_invocation.py                           |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/entrypoints/cli/services.py                                              |        7 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/arrays/token\_batch.py                                   |       39 |        0 |        4 |        0 |    100% |           |
| src/emblema/shared/adapters/in\_memory/artifact\_store.py                            |       28 |        0 |        6 |        0 |    100% |           |
| src/emblema/shared/adapters/in\_memory/clock.py                                      |        6 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/in\_memory/event\_publisher.py                           |        8 |        0 |        2 |        0 |    100% |           |
| src/emblema/shared/adapters/in\_memory/event\_subscriber.py                          |       10 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/in\_memory/id\_generator.py                              |        9 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/loaders/seeded\_shuffle\_sampler.py                      |       18 |        0 |        2 |        0 |    100% |           |
| src/emblema/shared/adapters/loaders/window\_dataset.py                               |       12 |        0 |        2 |        0 |    100% |           |
| src/emblema/shared/adapters/loaders/window\_loader.py                                |       20 |        0 |        2 |        0 |    100% |           |
| src/emblema/shared/adapters/storage/files.py                                         |       29 |        0 |        6 |        0 |    100% |           |
| src/emblema/shared/adapters/storage/layout.py                                        |       11 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/storage/local\_directory.py                              |       55 |        3 |       10 |        0 |     95% |     74-76 |
| src/emblema/shared/adapters/storage/s3.py                                            |       65 |        1 |       14 |        2 |     96% |80-\>85, 111 |
| src/emblema/shared/adapters/system/clock.py                                          |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/system/id\_generator.py                                  |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/tensors/token\_tensors.py                                |       30 |        0 |        2 |        0 |    100% |           |
| src/emblema/shared/adapters/windows/exceptions.py                                    |        9 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/windows/format.py                                        |       29 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/windows/window\_block.py                                 |       77 |        0 |       12 |        0 |    100% |           |
| src/emblema/shared/adapters/windows/window\_block\_writer.py                         |      118 |        4 |       26 |        2 |     96% |190, 205-207, 251-\>255 |
| src/emblema/shared/events/domain\_event.py                                           |        8 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/kernel/artifacts.py                                               |        8 |        0 |        2 |        0 |    100% |           |
| src/emblema/shared/kernel/checksums.py                                               |       42 |        0 |        8 |        0 |    100% |           |
| src/emblema/shared/kernel/compute.py                                                 |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/kernel/exceptions.py                                              |       10 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/kernel/identifiers.py                                             |       17 |        0 |        2 |        0 |    100% |           |
| src/emblema/shared/kernel/ordering.py                                                |       10 |        0 |        2 |        0 |    100% |           |
| src/emblema/shared/kernel/sampling.py                                                |        4 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/kernel/timestamps.py                                              |       11 |        0 |        4 |        0 |    100% |           |
| src/emblema/shared/kernel/tokens.py                                                  |       46 |        0 |       20 |        0 |    100% |           |
| src/emblema/shared/ports/artifact\_store.py                                          |       12 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/ports/clock.py                                                    |        3 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/ports/event\_publisher.py                                         |        3 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/ports/event\_subscriber.py                                        |        4 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/ports/exceptions.py                                               |        3 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/ports/id\_generator.py                                            |        3 |        0 |        0 |        0 |    100% |           |
| **TOTAL**                                                                            | **2290** |   **11** |  **428** |    **7** | **99%** |           |

32 empty files skipped.


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://github.com/lstasiak/emblema/raw/python-coverage-comment-action-data/badge.svg)](https://github.com/lstasiak/emblema/tree/python-coverage-comment-action-data)

This is the one to use if your repository is private or if you don't want to customize anything.



## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.