from repair_agent.repair_agent import RepairAgent
from repair_agent.models.test_document import TestDocument
from elasticsearch import Elasticsearch, helpers
import asyncio
from dotenv import load_dotenv

load_dotenv()

async def main():
    es = Elasticsearch("http://host.docker.internal:9200")
    docs = helpers.scan(
        es,
        index="test",
        query={
            "query":{
                "term":{
                    "status":"FAILED"
                }
            }
        }
    )

    state = {
        "test_documents": [
            TestDocument(
                testID=hit["_source"]["testID"],
                buildID=hit["_source"]["buildID"],
                repositoryUrl=hit["_source"]["repositoryUrl"],
                branchName=hit["_source"]["branchName"],
                jobName=hit["_source"]["jobName"],
                className=hit["_source"]["className"],
                methodName=hit["_source"]["methodName"],
                suiteName=hit["_source"]["suiteName"],
                status=hit["_source"]["status"],
                duration_test=hit["_source"]["duration_test"],
                stackTrace=hit["_source"]["stackTrace"],
                errorMessage=hit["_source"]["errorMessage"],
                timestampExecution=hit["_source"]["timestampExecution"],
                testCaseFilePath=hit["_source"]["testCaseFilePath"],
                moduleName=hit["_source"]["moduleName"],
                startLine=hit["_source"]["startLine"],
                endLine=hit["_source"]["endLine"],
                ownershipSource=hit["_source"]["ownershipSource"],
                confidenceScore=hit["_source"]["confidenceScore"],
                createdAt=hit["_source"]["createdAt"],
                lastModifiedAt=hit["_source"]["lastModifiedAt"],
                lastModifiedBy=hit["_source"]["lastModifiedBy"],
                currentCommitSha=hit["_source"]["currentCommitSha"],
            ) for hit in docs
        ]
    }

    agent = RepairAgent()

    result = await agent.ainvoke(state)

    print(result)


if __name__ == "__main__":
    asyncio.run(main())