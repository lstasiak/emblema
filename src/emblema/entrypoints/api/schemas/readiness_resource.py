from typing import Self

from pydantic import BaseModel

from emblema.entrypoints.api.readiness_report import ReadinessReport


class ReadinessResource(BaseModel):
    """Whether the process can do its job, and what each dependency answered."""

    status: str
    checks: dict[str, str]

    @classmethod
    def of(cls, report: ReadinessReport) -> Self:
        return cls(status="ready" if report.ready else "not ready", checks=dict(report.checks))
