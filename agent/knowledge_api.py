"""HTTP endpoints for the project-wide shared rule library."""

from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from agent.knowledge import KnowledgeBase, KnowledgeError, MAX_UPLOAD_BYTES


class DocumentUpdate(BaseModel):
    enabled: bool


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=8)
    region: str = Field(default="", max_length=200)
    disaster_type: str = Field(default="", max_length=100)


def knowledge_router(knowledge: KnowledgeBase) -> APIRouter:
    router = APIRouter(prefix="/api/knowledge", tags=["共享规则知识库"])

    def call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except KeyError as exc:
            raise HTTPException(404, "规则文档不存在") from exc
        except KnowledgeError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(503, "知识库数据库暂不可用，请检查后端日志") from exc

    @router.get("/status")
    def status():
        return call(knowledge.status)

    @router.get("/documents")
    def documents():
        return {"documents": call(knowledge.list_documents)}

    @router.post("/documents", status_code=202)
    def upload(file: UploadFile = File(...), source: str = Form("", max_length=500),
               version: str = Form("", max_length=100), effective_date: str = Form("", max_length=100),
               region: str = Form("", max_length=200), disaster_type: str = Form("", max_length=100)):
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        return call(knowledge.upload, file.filename or "", data, {
            "source": source.strip(), "version": version.strip(), "effective_date": effective_date.strip(),
            "region": region.strip(), "disaster_type": disaster_type.strip(),
        })

    @router.get("/documents/{document_id}")
    def detail(document_id: UUID):
        document = call(knowledge.get_document, str(document_id))
        with knowledge.database._conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT locator,content FROM knowledge_chunks WHERE document_id=%s ORDER BY ordinal LIMIT 5", (document_id,))
            preview = [{"locator": locator, "content": text} for locator, text in cur.fetchall()]
        return {"document": document, "preview": preview}

    @router.patch("/documents/{document_id}")
    def update(document_id: UUID, body: DocumentUpdate):
        return {"document": call(knowledge.update_document, str(document_id), enabled=body.enabled)}

    @router.post("/documents/{document_id}/retry", status_code=202)
    def retry(document_id: UUID):
        return {"document": call(knowledge.retry, str(document_id))}

    @router.delete("/documents/{document_id}")
    def delete(document_id: UUID):
        call(knowledge.delete, str(document_id))
        return {"ok": True}

    @router.get("/documents/{document_id}/file")
    def file(document_id: UUID):
        document = call(knowledge.get_document, str(document_id))
        path = knowledge.document_path(document)
        if not path.is_file():
            raise HTTPException(404, "原文文件不存在")
        return FileResponse(path, filename=document["filename"])

    @router.post("/search")
    def search(body: SearchRequest):
        return call(knowledge.search, body.query, body.top_k, body.region, body.disaster_type)

    return router
