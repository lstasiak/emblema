"""What a job may carry, and nothing else.

An alias rather than a class: this is the set of types that survive a broker, and a job's
arguments are read back by a process that has only the message. Stated once so that every port,
adapter and entry point that touches a job agrees on what may be put in one.
"""

type JobArgument = str | int | float | bool | None
