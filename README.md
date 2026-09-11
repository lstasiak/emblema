# Repository Coverage



| Name                                                               |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/emblema/catalog/adapters/in\_memory/corpus\_reader.py          |        8 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/adapters/in\_memory/corpus\_repository.py      |       16 |        0 |        4 |        0 |    100% |           |
| src/emblema/catalog/adapters/readers/cmapss.py                     |       54 |        0 |       18 |        0 |    100% |           |
| src/emblema/catalog/application/corpus\_version\_ref\_assembler.py |        6 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/application/register\_corpus.py                |       16 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/application/register\_corpus\_version.py       |       33 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/contracts/corpus\_version\_ref.py              |       23 |        0 |        8 |        0 |    100% |           |
| src/emblema/catalog/contracts/events.py                            |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/contracts/exceptions.py                        |        4 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/contracts/identifiers.py                       |        4 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/domain/channel\_schema.py                      |       26 |        0 |        6 |        0 |    100% |           |
| src/emblema/catalog/domain/corpus.py                               |       51 |        0 |       20 |        0 |    100% |           |
| src/emblema/catalog/domain/corpus\_content.py                      |       10 |        0 |        4 |        0 |    100% |           |
| src/emblema/catalog/domain/corpus\_description.py                  |        6 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/domain/corpus\_source.py                       |        9 |        0 |        4 |        0 |    100% |           |
| src/emblema/catalog/domain/corpus\_version.py                      |       46 |        0 |       12 |        0 |    100% |           |
| src/emblema/catalog/domain/exceptions.py                           |       36 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/domain/identifiers.py                          |        4 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/domain/licence.py                              |       10 |        0 |        4 |        0 |    100% |           |
| src/emblema/catalog/ports/corpus\_reader.py                        |        4 |        0 |        0 |        0 |    100% |           |
| src/emblema/catalog/ports/corpus\_repository.py                    |        6 |        0 |        0 |        0 |    100% |           |
| src/emblema/config/artifact\_store\_settings.py                    |        8 |        0 |        0 |        0 |    100% |           |
| src/emblema/config/settings.py                                     |       12 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/in\_memory/clock.py                    |        6 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/in\_memory/event\_publisher.py         |        8 |        0 |        2 |        0 |    100% |           |
| src/emblema/shared/adapters/in\_memory/event\_subscriber.py        |       10 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/in\_memory/id\_generator.py            |        9 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/storage/layout.py                      |        6 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/storage/local\_directory.py            |       45 |        3 |        6 |        0 |     94% |     61-63 |
| src/emblema/shared/adapters/storage/s3.py                          |       49 |        1 |       10 |        1 |     97% |        80 |
| src/emblema/shared/adapters/system/clock.py                        |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/adapters/system/id\_generator.py                |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/events/domain\_event.py                         |        8 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/kernel/artifacts.py                             |        8 |        0 |        2 |        0 |    100% |           |
| src/emblema/shared/kernel/checksums.py                             |       42 |        0 |        8 |        0 |    100% |           |
| src/emblema/shared/kernel/compute.py                               |        5 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/kernel/exceptions.py                            |        8 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/kernel/identifiers.py                           |       17 |        0 |        2 |        0 |    100% |           |
| src/emblema/shared/kernel/sampling.py                              |        4 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/kernel/timestamps.py                            |       11 |        0 |        4 |        0 |    100% |           |
| src/emblema/shared/ports/artifact\_store.py                        |        9 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/ports/clock.py                                  |        3 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/ports/event\_publisher.py                       |        3 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/ports/event\_subscriber.py                      |        4 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/ports/exceptions.py                             |        3 |        0 |        0 |        0 |    100% |           |
| src/emblema/shared/ports/id\_generator.py                          |        3 |        0 |        0 |        0 |    100% |           |
| **TOTAL**                                                          |  **668** |    **4** |  **114** |    **1** | **99%** |           |

21 empty files skipped.


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