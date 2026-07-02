from dataclasses import dataclass

@dataclass
class TestDocument:

    testID: str
    buildID: str
    repositoryUrl: str
    branchName: str
    jobName: str
    className: str
    methodName: str
    suiteName: str
    status: str
    duration_test: float
    stackTrace: str
    errorMessage: str
    timestampExecution: str
    testCaseFilePath: str
    moduleName: str
    startLine: int
    endLine: int
    ownershipSource: str
    confidenceScore: float
    createdAt: str
    lastModifiedAt: str
    lastModifiedBy: str
    currentCommitSha: str