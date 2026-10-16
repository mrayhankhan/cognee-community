"""Dataset isolation against an in-memory stand-in for a Moss project. No account needed."""

from types import SimpleNamespace
from uuid import uuid4

import pytest
from cognee.infrastructure.databases.vector import get_vectordb_config
from cognee_community_vector_adapter_moss import moss_adapter
from cognee_community_vector_adapter_moss.moss_adapter import IndexSchema, MossAdapter
from cognee_community_vector_adapter_moss.MossDatasetDatabaseHandler import (
    MossDatasetDatabaseHandler,
)
from contract_suite import FakeEmbeddingEngine

COLLECTION = "Entity_name"
PROJECT = "project-1"


class FakeMossClient:
    """Keeps indexes per project id, the way the Moss service does."""

    projects: dict[str, dict[str, dict]] = {}

    def __init__(self, project_id, project_key):
        assert "/" not in project_id
        self.indexes = self.projects.setdefault(project_id, {})

    async def list_indexes(self):
        return [SimpleNamespace(name=name) for name in self.indexes]

    async def create_index(self, name, docs):
        assert name not in self.indexes
        self.indexes[name] = {doc.id: doc for doc in docs}
        return SimpleNamespace(job_id="job")

    async def add_docs(self, name, docs, options):
        self.indexes[name].update({doc.id: doc for doc in docs})
        return SimpleNamespace(job_id="job")

    async def get_job_status(self, job_id):
        return SimpleNamespace(status=SimpleNamespace(value="completed"))

    async def load_index(self, name, auto_refresh=True):
        pass

    async def unload_index(self, name):
        pass

    async def delete_index(self, name):
        del self.indexes[name]

    async def delete_docs(self, name, doc_ids):
        for doc_id in doc_ids:
            self.indexes[name].pop(doc_id, None)

    async def get_docs(self, name, options):
        docs = self.indexes.get(name, {})
        return [docs[doc_id] for doc_id in options.doc_ids if doc_id in docs]

    async def query(self, name, query_text, options):
        conditions = options.filter["$and"]
        return SimpleNamespace(
            docs=[
                SimpleNamespace(id=doc.id, metadata=doc.metadata, score=0.1)
                for doc in self.indexes.get(name, {}).values()
                if all(doc.metadata[c["field"]] == c["condition"]["$eq"] for c in conditions)
            ]
        )


@pytest.fixture(autouse=True)
def fake_moss(monkeypatch):
    FakeMossClient.projects = {}
    monkeypatch.setattr(moss_adapter, "MossClient", FakeMossClient)
    config = get_vectordb_config()
    monkeypatch.setattr(config, "vector_db_provider", "moss")
    monkeypatch.setattr(config, "vector_db_name", PROJECT)
    monkeypatch.setattr(config, "vector_db_key", "key")


def adapter(database_name):
    return MossAdapter(
        url=None, api_key="key", embedding_engine=FakeEmbeddingEngine(), database_name=database_name
    )


async def dataset_adapters():
    names = [
        (await MossDatasetDatabaseHandler.create_dataset(uuid4(), None))["vector_database_name"]
        for _ in range(2)
    ]
    return [adapter(name) for name in names]


@pytest.mark.asyncio
async def test_handler_gives_each_dataset_its_own_name_in_the_project():
    dataset_id = uuid4()
    info = await MossDatasetDatabaseHandler.create_dataset(dataset_id, None)
    other = await MossDatasetDatabaseHandler.create_dataset(uuid4(), None)

    assert info["vector_database_name"] == f"{PROJECT}/{dataset_id}"
    assert other["vector_database_name"] != info["vector_database_name"]


@pytest.mark.asyncio
@pytest.mark.parametrize("delete_method", ["delete_data_points", "prune"])
async def test_shared_entity_survives_other_dataset_write_and_deletion(delete_method):
    a, b = await dataset_adapters()
    point_id = uuid4()
    for engine, text in ((a, "Alice in A"), (b, "Alice in B")):
        await engine.create_data_points(COLLECTION, [IndexSchema(id=point_id, text=text)])

    for engine, text in ((a, "Alice in A"), (b, "Alice in B")):
        records = await engine.retrieve(COLLECTION, [str(point_id)])
        assert [record.payload["text"] for record in records] == [text]
        hits = await engine.search(COLLECTION, query_text="Alice", include_payload=True)
        assert [hit.payload["text"] for hit in hits] == [text]

    if delete_method == "prune":
        await a.prune()
    else:
        await a.delete_data_points(COLLECTION, [str(point_id)])
    assert await a.retrieve(COLLECTION, [str(point_id)]) == []

    # A fresh adapter, as cognee builds per request, still finds B's data.
    remaining = await adapter(b.database_name).retrieve(COLLECTION, [str(point_id)])
    assert [record.payload["text"] for record in remaining] == ["Alice in B"]


@pytest.mark.asyncio
async def test_deleting_a_dataset_keeps_the_other_datasets(monkeypatch):
    import cognee_community_vector_adapter_moss.MossDatasetDatabaseHandler as handler_module

    monkeypatch.setattr(
        handler_module,
        "create_vector_engine",
        lambda vector_db_provider, vector_db_url, vector_db_key, vector_db_name: adapter(
            vector_db_name
        ),
    )
    a, b = await dataset_adapters()
    await a.create_data_points(COLLECTION, [IndexSchema(id=uuid4(), text="kept")])
    await b.create_data_points(COLLECTION, [IndexSchema(id=uuid4(), text="deleted")])

    await MossDatasetDatabaseHandler.delete_dataset(
        SimpleNamespace(
            vector_database_provider="moss",
            vector_database_url=None,
            vector_database_key="key",
            vector_database_name=b.database_name,
        )
    )

    hits = await adapter(a.database_name).search(COLLECTION, query_text="x", include_payload=True)
    assert [hit.payload["text"] for hit in hits] == ["kept"]
    assert await adapter(b.database_name).search(COLLECTION, query_text="x") == []


@pytest.mark.asyncio
async def test_single_user_mode_keeps_the_shared_legacy_index():
    a, _ = await dataset_adapters()
    await a.create_data_points(COLLECTION, [IndexSchema(id=uuid4(), text="dataset doc")])
    legacy = adapter(PROJECT)
    await legacy.create_data_points(COLLECTION, [IndexSchema(id=uuid4(), text="legacy doc")])

    indexes = FakeMossClient.projects[PROJECT]
    assert sorted(name.split("-")[1] for name in indexes) == ["dataset", "index"]
    hits = await adapter(PROJECT).search(COLLECTION, query_text="x", include_payload=True)
    assert [hit.payload["text"] for hit in hits] == ["legacy doc"]

    await legacy.prune()
    assert list(indexes) == [a._index_name]
