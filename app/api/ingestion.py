from pathlib import Path
import shutil
import uuid

from fastapi import APIRouter, Depends, File,Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document
from app.db.session import get_db
from app.services.chunking import extract_text, chunk_text
from app.services.embeddings import embed_chunks
from app.services.vectorstore import store_in_qdrant


router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload")
def upload_document(
    file: UploadFile = File(...),
    strategy: str = Form("recursive"),
    db: Session = Depends(get_db),
):
    try:
        # 1. Save uploaded file temporarily
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)

        temp_filename = f"{uuid.uuid4()}_{file.filename}"
        temp_path = upload_dir / temp_filename

        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Extract text
        text = extract_text(str(temp_path))

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="The uploaded document is empty.",
            )

        # 3. Split text into chunks
        chunks = chunk_text(text,strategy=strategy)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No chunks could be created from the document.",
            )

        # 4. Generate embeddings
        embeddings = embed_chunks(chunks)

        # 5. Store embeddings + chunks in Qdrant
        store_in_qdrant(chunks, embeddings)

        # 6. Save document metadata to SQLite
        document = Document(
            filename=file.filename,
            content=text,
            strategy=strategy,
        )

        db.add(document)
        db.flush()

        # 7. Save individual chunks to SQLite
        for chunk_text_value in chunks:
            chunk = Chunk(
                document_id=document.id,
                content=chunk_text_value,
            )
            db.add(chunk)

        db.commit()
        db.refresh(document)

        # 8. Remove temporary file
        temp_path.unlink(missing_ok=True)

        return {
            "doc_id": document.id,
            "chunk_count": len(chunks),
            "strategy": strategy,
        }

    except HTTPException:
        raise

    except Exception as e:
        db.rollback()

        # Try to remove temporary file if something failed
        if "temp_path" in locals():
            temp_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
