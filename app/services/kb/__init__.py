"""P2 KB substrate service seams (T26, V-KB1).

The four sub-modules are the side-effecting boundaries between the
substrate tables and the outside world:

- ``embeddings`` — OpenAI embeddings API → ``content_embeddings``
- ``similarity`` — cosine over node embeddings → ``concept_edges`` (V-E2)
- ``pdf_ingest`` — pdfplumber → ``pdf_sources`` + ``atomic_facts`` (V-KB1)
- ``notion``    — notion-client → ``notion_pages`` (V-N1, V-N2)

Per V16 the OpenAI + Notion clients are injected and mocked at the SDK
boundary in tests; never construct a real ``AsyncOpenAI`` /
``notion_client.AsyncClient`` inside the seam.
"""
