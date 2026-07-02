from http.client import responses

from repair_agent.repair_agent import RepairAgent
from repair_agent.models.test_document import TestDocument
from elasticsearch import Elasticsearch
import asyncio
from dotenv import load_dotenv

load_dotenv()

async def main():
    es = Elasticsearch("http://host.docker.internal:9200")
    response = es.get(index="test", id="FuncS1Test#testFunc1_033")

    test_doc = response["_source"]

    print(test_doc)
    # test_doc = TestDocument(
    #     {
            # "testID": "FuncS1Test#testFunc1_033",
            # "buildID": "15323CC3-5FA2-4DF1-8CFD-CB749844708E",
            # "repositoryUrl": "https://github.com/Aman1406-gupta/random-project",
            # "branchName": "HEAD",
            # "jobName": "Random-Project",
            # "className": "com.sprinklr.randomproject.funcs.FuncS1Test",
            # "methodName": "testFunc1_033",
            # "suiteName": "Suite1",
            # "status": "FAILED",
            # "duration_test": 0.003,
            # "stackTrace": """org.opentest4j.AssertionFailedError: expected: <17> but was: <16>
            # at app//org.junit.jupiter.api.AssertionFailureBuilder.build(AssertionFailureBuilder.java:151)
            # at app//org.junit.jupiter.api.AssertionFailureBuilder.buildAndThrow(AssertionFailureBuilder.java:132)
            # at app//org.junit.jupiter.api.AssertEquals.failNotEqual(AssertEquals.java:197)
            # at app//org.junit.jupiter.api.AssertEquals.assertEquals(AssertEquals.java:150)
            # at app//org.junit.jupiter.api.AssertEquals.assertEquals(AssertEquals.java:145)
            # at app//org.junit.jupiter.api.Assertions.assertEquals(Assertions.java:531)
            # at app//com.sprinklr.randomproject.funcs.FuncS1Test.testFunc1_033(FuncS1Test.java:322)
            # at java.base@21.0.11/java.lang.reflect.Method.invoke(Method.java:580)
            # at java.base@21.0.11/java.util.ArrayList.forEach(ArrayList.java:1596)
            # at java.base@21.0.11/java.util.ArrayList.forEach(ArrayList.java:1596)
            # """,
            # "errorMessage": "org.opentest4j.AssertionFailedError: expected: <17> but was: <16>",
            # "timestampExecution": "2026-06-20T19:51:00.852Z",
            # "testCaseFilePath": "src/test/java/com/sprinklr/randomproject/funcs/FuncS1Test.java",
            # "moduleName": "root",
            # "startLine": 317,
            # "endLine": 323,
            # "ownershipSource": "GIT_BLAME",
            # "confidenceScore": 1,
            # "createdAt": "2026-06-08T13:06:00.000Z",
            # "lastModifiedAt": "2026-06-08T13:06:00.000Z",
            # "lastModifiedBy": "Owner Three",
            # "currentCommitSha": "661762feb994f6ce098e2cbe6ab1f32687c8fff2"
    #     }
    # )

    state = {
        "test_document": test_doc,
    }

    agent = RepairAgent()

    result = await agent.ainvoke(state)

    print(result)

    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())