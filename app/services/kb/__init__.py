"""P2 KB substrate service seams (T26, V-KB1) + P3 recall layer (T28, V-L3).

The substrate sub-modules are the side-effecting boundaries between the
substrate tables and the outside world:

- ``embeddings`` — OpenAI embeddings API → ``content_embeddings``
- ``similarity`` — cosine over node embeddings → ``concept_edges`` (V-E2)
- ``pdf_ingest`` — PyMuPDF render → OpenAI vision transcription → OpenAI
  structured-output fact extraction → ``pdf_sources`` + ``atomic_facts``
  (V-KB1, V-KB3, V-KB4)
- ``notion``    — notion-client → ``notion_pages`` (V-N1, V-N2)

Read-side:

- ``recall`` — candidate retrieval over ``content_embeddings`` +
  ``concept_edges`` + prior calibrated ``question_tags`` for V-L3
  constrained tagging prompts.

Per V16 the OpenAI + Notion clients are injected and mocked at the SDK
boundary in tests; never construct a real ``AsyncOpenAI`` /
``notion_client.AsyncClient`` inside the seam.
"""
