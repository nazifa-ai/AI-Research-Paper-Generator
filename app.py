import streamlit as st
import fitz
import json
import re
import io
import zipfile
from typing import Optional
from pydantic import BaseModel, Field
from jinja2 import Template
import google.generativeai as genai

# ── Models ────────────────────────────────────────────────────────────────────

class Author(BaseModel):
    name: str
    affiliation: str = ""
    email: str = ""

class Figure(BaseModel):
    caption: str
    image_bytes: bytes = b""
    page: int = 0

class Table(BaseModel):
    caption: str
    content: str

class Citation(BaseModel):
    key: str
    title: str
    authors: list[str] = []
    year: str = ""
    journal: str = ""
    doi: str = ""
    volume: str = ""
    pages: str = ""

class Section(BaseModel):
    heading: str
    content: str

class ExtractedDocument(BaseModel):
    id: str
    filename: str
    title: str = ""
    authors: list[str] = []
    year: str = ""
    journal: str = ""
    doi: str = ""
    abstract: str = ""
    sections: dict[str, str] = {}
    figures: list[Figure] = []
    tables: list[Table] = []
    extraction_ok: bool = True
    extraction_warning: str = ""

class KnowledgeBase(BaseModel):
    themes: list[str] = []
    methods: list[str] = []
    datasets: list[str] = []
    findings: list[str] = []
    limitations: list[str] = []
    future_work: list[str] = []
    citations: list[Citation] = []

class PaperConfig(BaseModel):
    title: str = "Research Paper"
    authors: list[str] = []
    affiliations: list[str] = []
    corresponding_author: str = ""
    email: str = ""
    keywords: list[str] = []
    paper_type: str = "journal"
    template: str = "ieee"
    layout: str = "double"
    target_pages: int = 8
    target_words: int = 6000

class EvidenceScore(BaseModel):
    section: str
    paragraph_idx: int
    coverage_pct: float
    sources_used: int
    unsupported: bool

class CitationMap(BaseModel):
    mapping: dict[str, dict[int, list[str]]] = {}

class PaperProject(BaseModel):
    config: PaperConfig
    kb: KnowledgeBase
    outline: dict = {}
    sections: dict[str, str] = {}
    citation_map: CitationMap = CitationMap()
    evidence_scores: list[EvidenceScore] = []
    bibliography: list[Citation] = []
    latex_files: dict[str, str] = {}

# ── Session state init ────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "uploaded_pdfs": [],
        "uploaded_images": [],
        "extracted_documents": [],
        "knowledge_base": None,
        "paper_config": PaperConfig(),
        "outline": {},
        "generated_sections": {},
        "citation_map": CitationMap(),
        "evidence_scores": [],
        "bibliography": [],
        "latex_files": {},
        "current_step": 1,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ── Gemini helper ─────────────────────────────────────────────────────────────

def call_gemini(prompt: str, system: str = "", json_mode: bool = False) -> str | dict:
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        if not api_key:
            st.error("GEMINI_API_KEY not found in Streamlit secrets.")
            return {} if json_mode else ""
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            "gemini-2.0-flash-lite",
            system_instruction=system if system else None,
        )
        full_prompt = prompt
        if json_mode:
            full_prompt += "\n\nRespond with valid JSON only. No markdown fences, no preamble."
        resp = model.generate_content(full_prompt)
        text = resp.text.strip()
        if json_mode:
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                st.warning("Gemini returned malformed JSON — attempting partial parse.")
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group())
                    except Exception:
                        pass
                st.error("Could not parse Gemini JSON response.")
                return {}
        return text
    except Exception as e:
        st.error(f"Gemini API error: {e}")
        return {} if json_mode else ""

# ── PDF extraction ────────────────────────────────────────────────────────────

SECTION_HEADINGS = re.compile(
    r"^\s*(abstract|introduction|related work|background|literature review|"
    r"methodology|methods|materials and methods|experimental setup|results|"
    r"discussion|conclusion|conclusions|future work|references|acknowledgements?)\s*$",
    re.IGNORECASE,
)

def extract_doi(text: str) -> str:
    m = re.search(r"10\.\d{4,9}/[^\s\"'<>]+", text)
    return m.group() if m else ""

def extract_year(text: str) -> str:
    m = re.search(r"\b(19|20)\d{2}\b", text)
    return m.group() if m else ""

def extract_pdf(file_bytes: bytes, filename: str, doc_id: str) -> ExtractedDocument:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        meta = doc.metadata or {}
        first_page_text = doc[0].get_text() if len(doc) > 0 else ""

        title = meta.get("title", "") or filename.replace(".pdf", "")
        author_raw = meta.get("author", "")
        authors = [a.strip() for a in re.split(r"[;,]", author_raw) if a.strip()] if author_raw else []
        year = extract_year(first_page_text) or meta.get("creationDate", "")[:4]
        doi = extract_doi(first_page_text)
        journal = ""

        full_text = "\n".join(page.get_text() for page in doc)
        sections: dict[str, str] = {}
        current_section = "Other"
        buffer: list[str] = []

        for line in full_text.split("\n"):
            if SECTION_HEADINGS.match(line.strip()):
                if buffer:
                    sections[current_section] = sections.get(current_section, "") + "\n".join(buffer) + "\n"
                    buffer = []
                current_section = line.strip().title()
            else:
                buffer.append(line)
        if buffer:
            sections[current_section] = sections.get(current_section, "") + "\n".join(buffer)

        abstract = sections.pop("Abstract", "")
        if not abstract:
            m = re.search(r"(?i)abstract[.\n\s]*(.{100,1500}?)(?:introduction|keywords|1\.)", full_text, re.DOTALL)
            abstract = m.group(1).strip() if m else ""

        figures: list[Figure] = []
        for page in doc:
            for img in page.get_images(full=True):
                xref = img[0]
                try:
                    img_bytes = doc.extract_image(xref)["image"]
                    figures.append(Figure(caption=f"Figure from page {page.number + 1}", image_bytes=img_bytes, page=page.number))
                except Exception:
                    pass

        return ExtractedDocument(
            id=doc_id,
            filename=filename,
            title=title,
            authors=authors,
            year=year,
            journal=journal,
            doi=doi,
            abstract=abstract,
            sections=sections,
            figures=figures,
        )
    except Exception as e:
        return ExtractedDocument(
            id=doc_id,
            filename=filename,
            extraction_ok=False,
            extraction_warning=str(e),
        )

# ── Knowledge base ────────────────────────────────────────────────────────────

def build_kb_for_doc(doc: ExtractedDocument) -> KnowledgeBase:
    snippet = f"Title: {doc.title}\nAbstract: {doc.abstract[:800]}\n"
    for sec, txt in list(doc.sections.items())[:6]:
        snippet += f"\n{sec}:\n{txt[:600]}\n"
    result = call_gemini(
        prompt=snippet,
        system=(
            "Extract and summarize into JSON with keys: themes, methods, datasets, findings, limitations, future_work. "
            "Each key is a list of short strings. Summarize only. Do not generate manuscript prose. "
            "Do not invent facts not present in the source text."
        ),
        json_mode=True,
    )
    if not result:
        return KnowledgeBase()
    return KnowledgeBase(
        themes=result.get("themes", []),
        methods=result.get("methods", []),
        datasets=result.get("datasets", []),
        findings=result.get("findings", []),
        limitations=result.get("limitations", []),
        future_work=result.get("future_work", []),
        citations=[Citation(
            key=doc.id,
            title=doc.title,
            authors=doc.authors,
            year=doc.year,
            journal=doc.journal,
            doi=doc.doi,
        )],
    )

def merge_kbs(kbs: list[KnowledgeBase]) -> KnowledgeBase:
    def dedup(lst: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in lst:
            norm = item.lower().strip()
            if norm not in seen:
                seen.add(norm)
                out.append(item)
        return out

    merged = KnowledgeBase()
    for kb in kbs:
        merged.themes += kb.themes
        merged.methods += kb.methods
        merged.datasets += kb.datasets
        merged.findings += kb.findings
        merged.limitations += kb.limitations
        merged.future_work += kb.future_work
        merged.citations += kb.citations
    merged.themes = dedup(merged.themes)
    merged.methods = dedup(merged.methods)
    merged.datasets = dedup(merged.datasets)
    merged.findings = dedup(merged.findings)
    merged.limitations = dedup(merged.limitations)
    merged.future_work = dedup(merged.future_work)
    return merged

# ── Outline ───────────────────────────────────────────────────────────────────

def generate_outline(kb: KnowledgeBase, config: PaperConfig) -> dict:
    kb_summary = (
        f"Themes: {', '.join(kb.themes[:8])}\n"
        f"Methods: {', '.join(kb.methods[:8])}\n"
        f"Findings: {', '.join(kb.findings[:8])}\n"
        f"Limitations: {', '.join(kb.limitations[:5])}\n"
        f"Future work: {', '.join(kb.future_work[:5])}\n"
    )
    prompt = (
        f"Paper title: {config.title}\nKeywords: {', '.join(config.keywords)}\n"
        f"Target length: {config.target_words} words\n\n{kb_summary}\n\n"
        "Generate a JSON outline with keys: abstract, introduction, literature_review, methodology, results, discussion, conclusion. "
        "Each key maps to a list of 3-5 bullet strings. Outline only, no full prose."
    )
    return call_gemini(prompt=prompt, json_mode=True)

# ── Section generators ────────────────────────────────────────────────────────

SYSTEM_WRITER = (
    "You are an academic writer. Write in formal scientific prose. "
    "Return JSON with exactly two keys: 'content' (string, the section text) and "
    "'citations_used' (list of source id strings from the provided sources)."
)

def _section_call(prompt: str) -> tuple[str, list[str]]:
    result = call_gemini(prompt=prompt, system=SYSTEM_WRITER, json_mode=True)
    if not result:
        return "", []
    return result.get("content", ""), result.get("citations_used", [])

def generate_abstract(kb: KnowledgeBase, outline: dict, config: PaperConfig) -> tuple[str, list[str]]:
    bullets = "\n".join(f"- {b}" for b in outline.get("abstract", []))
    source_ids = [c.key for c in kb.citations]
    return _section_call(
        f"Write the Abstract section.\nTitle: {config.title}\nBullets: {bullets}\n"
        f"Available source ids: {source_ids}\nWord target: ~200"
    )

def generate_introduction(kb: KnowledgeBase, outline: dict, config: PaperConfig) -> tuple[str, list[str]]:
    bullets = "\n".join(f"- {b}" for b in outline.get("introduction", []))
    themes = ", ".join(kb.themes[:6])
    source_ids = [c.key for c in kb.citations]
    return _section_call(
        f"Write the Introduction.\nTitle: {config.title}\nThemes: {themes}\nBullets: {bullets}\n"
        f"Available source ids: {source_ids}\nWord target: ~500"
    )

def generate_literature_review(kb: KnowledgeBase, outline: dict) -> tuple[str, list[str]]:
    bullets = "\n".join(f"- {b}" for b in outline.get("literature_review", []))
    cits = "\n".join(f"{c.key}: {c.title} ({c.year})" for c in kb.citations)
    return _section_call(
        f"Write the Literature Review.\nSources:\n{cits}\nBullets: {bullets}\n"
        f"Available source ids: {[c.key for c in kb.citations]}\nWord target: ~800"
    )

def generate_methodology(kb: KnowledgeBase, outline: dict, docs: list[ExtractedDocument]) -> tuple[str, list[str]]:
    raw = "\n\n".join(
        f"[{d.id}] {d.sections.get('Methodology', d.sections.get('Methods', d.sections.get('Materials And Methods', '')))[:1000]}"
        for d in docs
        if any(k in d.sections for k in ["Methodology", "Methods", "Materials And Methods"])
    )
    if not raw.strip():
        st.warning("⚠️ No methodology sections found in uploaded PDFs. Methodology section may be incomplete.")
    bullets = "\n".join(f"- {b}" for b in outline.get("methodology", []))
    return _section_call(
        f"Write the Methodology section.\nExtracted methodology text:\n{raw[:3000]}\nBullets: {bullets}\n"
        "Use only the provided extracted data. Do not invent methods, datasets, or metrics.\n"
        f"Available source ids: {[d.id for d in docs]}\nWord target: ~700"
    )

def generate_results(kb: KnowledgeBase, outline: dict, docs: list[ExtractedDocument]) -> tuple[str, list[str]]:
    raw = "\n\n".join(
        f"[{d.id}] {d.sections.get('Results', '')[:1000]}"
        for d in docs
        if "Results" in d.sections
    )
    if not raw.strip():
        st.warning("⚠️ No results sections found in uploaded PDFs. Results section may be incomplete.")
    bullets = "\n".join(f"- {b}" for b in outline.get("results", []))
    return _section_call(
        f"Write the Results section.\nExtracted results text:\n{raw[:3000]}\nBullets: {bullets}\n"
        "Use only the provided extracted data. Do not invent results, metrics, or datasets.\n"
        f"Available source ids: {[d.id for d in docs]}\nWord target: ~700"
    )

def generate_discussion(kb: KnowledgeBase, outline: dict) -> tuple[str, list[str]]:
    bullets = "\n".join(f"- {b}" for b in outline.get("discussion", []))
    findings = ", ".join(kb.findings[:6])
    limitations = ", ".join(kb.limitations[:4])
    return _section_call(
        f"Write the Discussion.\nFindings: {findings}\nLimitations: {limitations}\nBullets: {bullets}\n"
        f"Available source ids: {[c.key for c in kb.citations]}\nWord target: ~700"
    )

def generate_conclusion(kb: KnowledgeBase, outline: dict, config: PaperConfig) -> tuple[str, list[str]]:
    bullets = "\n".join(f"- {b}" for b in outline.get("conclusion", []))
    future = ", ".join(kb.future_work[:4])
    return _section_call(
        f"Write the Conclusion.\nTitle: {config.title}\nFuture work: {future}\nBullets: {bullets}\n"
        f"Available source ids: {[c.key for c in kb.citations]}\nWord target: ~400"
    )

# ── Citation engine ───────────────────────────────────────────────────────────

def build_citation_map(generated: dict[str, list[str]]) -> CitationMap:
    cm = CitationMap()
    for section, (text, cits) in generated.items():
        paragraphs = text.split("\n\n")
        cm.mapping[section] = {}
        for idx, _ in enumerate(paragraphs):
            cm.mapping[section][idx] = cits
    return cm

def make_bibtex(cit: Citation) -> str:
    if not cit.doi and not cit.year:
        key = f"{cit.key}_MISSING"
        prefix = "% WARNING: incomplete metadata\n"
    else:
        key = cit.key
        prefix = ""
    authors = " and ".join(cit.authors) if cit.authors else "Unknown"
    return (
        f"{prefix}@article{{{key},\n"
        f"  title = {{{cit.title}}},\n"
        f"  author = {{{authors}}},\n"
        f"  year = {{{cit.year or 'XXXX'}}},\n"
        f"  journal = {{{cit.journal or 'Unknown Journal'}}},\n"
        + (f"  doi = {{{cit.doi}}},\n" if cit.doi else "")
        + f"  volume = {{{cit.volume}}},\n"
        f"  pages = {{{cit.pages}}},\n"
        f"}}\n"
    )

def dedup_citations(cits: list[Citation]) -> list[Citation]:
    seen: set[str] = set()
    out: list[Citation] = []
    for c in cits:
        key = (c.doi or c.title.lower().strip())
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out

# ── Evidence coverage ─────────────────────────────────────────────────────────

def compute_evidence(sections: dict[str, str], citation_map: CitationMap) -> list[EvidenceScore]:
    scores: list[EvidenceScore] = []
    for section, text in sections.items():
        paragraphs = text.split("\n\n")
        for idx, para in enumerate(paragraphs):
            if not para.strip():
                continue
            mapped_cits = citation_map.mapping.get(section, {}).get(idx, [])
            sources_used = len(mapped_cits)
            sentences = [s for s in re.split(r"[.!?]", para) if s.strip()]
            total_s = max(len(sentences), 1)
            coverage = min(sources_used / total_s, 1.0)
            scores.append(EvidenceScore(
                section=section,
                paragraph_idx=idx,
                coverage_pct=round(coverage * 100, 1),
                sources_used=sources_used,
                unsupported=(sources_used == 0),
            ))
    return scores

# ── LaTeX templates ───────────────────────────────────────────────────────────

IEEE_TEMPLATE = Template(r"""
\documentclass[{% if layout == 'double' %}conference{% else %}journal{% endif %}]{IEEEtran}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\begin{document}
\title{ {{- title -}} }
\author{
{% for author in authors %}\IEEEauthorblockN{ {{- author -}} }
{% endfor %}}
\maketitle
\begin{abstract}
\input{sections/abstract}
\end{abstract}
\begin{IEEEkeywords}
{{ keywords | join(', ') }}
\end{IEEEkeywords}
\input{sections/introduction}
\input{sections/literature_review}
\input{sections/methodology}
\input{sections/results}
\input{sections/discussion}
\input{sections/conclusion}
\bibliographystyle{IEEEtran}
\bibliography{references}
\end{document}
""")

GENERIC_TEMPLATE = Template(r"""
\documentclass[12pt{% if layout == 'double' %},twocolumn{% endif %}]{article}
\usepackage{cite}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{hyperref}
\title{ {{- title -}} }
\author{ {{ authors | join(' \and ') }} }
\date{ {{ year }} }
\begin{document}
\maketitle
\begin{abstract}
\input{sections/abstract}
\end{abstract}
\textbf{Keywords:} {{ keywords | join('; ') }}
\input{sections/introduction}
\input{sections/literature_review}
\input{sections/methodology}
\input{sections/results}
\input{sections/discussion}
\input{sections/conclusion}
\bibliographystyle{plain}
\bibliography{references}
\end{document}
""")

ACM_STUB = r"""\documentclass{acmart}
% ACM template stub — populate sections as per IEEE/generic pattern
\begin{document}
\maketitle
% ... section inputs ...
\end{document}
"""

SPRINGER_STUB = r"""\documentclass{svjour3}
% Springer template stub — populate sections as per IEEE/generic pattern
\begin{document}
\maketitle
% ... section inputs ...
\end{document}
"""

def render_main_tex(config: PaperConfig, year: str = "") -> str:
    tmpl = {"ieee": IEEE_TEMPLATE, "generic": GENERIC_TEMPLATE}.get(config.template, GENERIC_TEMPLATE)
    return tmpl.render(
        title=config.title,
        authors=config.authors,
        keywords=config.keywords,
        layout=config.layout,
        year=year,
    )

def build_latex_files(config: PaperConfig, sections: dict[str, str], bibliography: list[Citation]) -> dict[str, str]:
    files: dict[str, str] = {}
    files["main.tex"] = render_main_tex(config)
    for sec_name, text in sections.items():
        label = sec_name.lower().replace(" ", "_")
        header = f"\\section{{{sec_name.title()}}}\n" if sec_name != "abstract" else ""
        files[f"sections/{label}.tex"] = header + text
    files["references.bib"] = "\n".join(make_bibtex(c) for c in bibliography)
    readme = (
        "# Research Paper LaTeX Package\n\n"
        f"Title: {config.title}\n"
        f"Template: {config.template.upper()}\n"
        f"Layout: {config.layout} column\n\n"
        "## Compile\n```\npdflatex main.tex\nbibtex main\npdflatex main.tex\npdflatex main.tex\n```\n"
    )
    files["README.md"] = readme
    return files

# ── ZIP export ────────────────────────────────────────────────────────────────

def build_zip(latex_files: dict[str, str], docs: list[ExtractedDocument]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, content in latex_files.items():
            zf.writestr(path, content)
        for doc in docs:
            for i, fig in enumerate(doc.figures):
                if fig.image_bytes:
                    zf.writestr(f"figures/{doc.id}_fig{i}.png", fig.image_bytes)
    buf.seek(0)
    return buf.read()

# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.step-bar { display: flex; gap: 4px; margin-bottom: 1.5rem; }
.step-item {
    flex: 1; padding: 6px 2px; border-radius: 6px; text-align: center;
    font-size: 11px; font-weight: 500; cursor: default;
    background: #f0f0f0; color: #888; border: 1px solid #e0e0e0;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.step-item.active { background: #1a1a2e; color: #e8e0ff; border-color: #1a1a2e; }
.step-item.done { background: #e8f5e9; color: #2e7d32; border-color: #a5d6a7; }

.card {
    background: #fafafa; border: 1px solid #e8e8e8; border-radius: 10px;
    padding: 1.2rem 1.4rem; margin-bottom: 1rem;
}
.section-label {
    font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: .08em; color: #888; margin-bottom: .4rem;
}
.cov-green { color: #2e7d32; font-weight: 600; }
.cov-amber { color: #e65100; font-weight: 600; }
.cov-red { color: #b71c1c; font-weight: 600; }

.stTextArea textarea { font-family: 'DM Mono', monospace; font-size: 13px; }
h1 { font-size: 1.6rem; font-weight: 600; color: #1a1a2e; }
h2 { font-size: 1.2rem; font-weight: 600; color: #1a1a2e; }
</style>
"""

STEPS = [
    "Upload", "Extract", "Knowledge Base", "Configure",
    "Outline", "Generate", "Citations", "Evidence", "Preview", "Export"
]

def render_step_bar(current: int):
    items = ""
    for i, label in enumerate(STEPS, 1):
        cls = "active" if i == current else ("done" if i < current else "step-item")
        # always include step-item base class
        full_cls = f"step-item {cls}" if cls != "step-item" else "step-item"
        items += f'<div class="{full_cls}">{i}. {label}</div>'
    st.markdown(f'<div class="step-bar">{items}</div>', unsafe_allow_html=True)

def nav_buttons(back_step: int | None = None, next_step: int | None = None, next_label: str = "Next →"):
    cols = st.columns([1, 4, 1])
    if back_step is not None:
        with cols[0]:
            if st.button("← Back"):
                st.session_state.current_step = back_step
                st.rerun()
    if next_step is not None:
        with cols[2]:
            if st.button(next_label, type="primary"):
                st.session_state.current_step = next_step
                st.rerun()

# ── Steps ─────────────────────────────────────────────────────────────────────

def step_upload():
    st.markdown("## 1 · Upload PDFs")
    st.caption("Upload the source research papers you want to synthesize.")
    uploaded = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    if uploaded:
        st.session_state.uploaded_pdfs = uploaded
        st.success(f"{len(uploaded)} file(s) ready.")

    images = st.file_uploader("Optional: upload figures/images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if images:
        st.session_state.uploaded_images = images

    nav_buttons(next_step=2 if st.session_state.uploaded_pdfs else None, next_label="Extract →")
    if not st.session_state.uploaded_pdfs:
        st.info("Upload at least one PDF to continue.")

def step_extract():
    st.markdown("## 2 · Extract Documents")
    if not st.session_state.uploaded_pdfs:
        st.warning("No PDFs uploaded. Go back to step 1.")
        nav_buttons(back_step=1)
        return

    if st.button("Run Extraction", type="primary"):
        docs = []
        for i, f in enumerate(st.session_state.uploaded_pdfs):
            with st.spinner(f"Extracting {f.name}…"):
                doc = extract_pdf(f.read(), f.name, f"doc{i}")
            docs.append(doc)

        st.session_state.extracted_documents = docs

    if st.session_state.extracted_documents:
        for doc in st.session_state.extracted_documents:
            icon = "✅" if doc.extraction_ok else "❌"
            with st.expander(f"{icon} {doc.filename}"):
                if doc.extraction_ok:
                    st.write(f"**Title:** {doc.title or '(not detected)'}")
                    st.write(f"**Authors:** {', '.join(doc.authors) or '(not detected)'}")
                    st.write(f"**Year:** {doc.year or '(not detected)'}")
                    st.write(f"**DOI:** {doc.doi or '(not detected)'}")
                    st.write(f"**Sections found:** {', '.join(doc.sections.keys())}")
                    st.write(f"**Figures:** {len(doc.figures)}")
                    if doc.abstract:
                        st.text_area("Abstract preview", doc.abstract[:400], height=80, disabled=True)
                else:
                    st.error(f"Extraction failed: {doc.extraction_warning}")

        nav_buttons(back_step=1, next_step=3, next_label="Build Knowledge Base →")
    else:
        nav_buttons(back_step=1)

def step_kb():
    st.markdown("## 3 · Build Knowledge Base")
    docs = st.session_state.extracted_documents
    if not docs:
        st.warning("No extracted documents. Go back.")
        nav_buttons(back_step=2)
        return

    if st.button("Build Knowledge Base", type="primary"):
        kbs = []
        progress = st.progress(0)
        for i, doc in enumerate(docs):
            if not doc.extraction_ok:
                continue
            with st.spinner(f"Summarizing {doc.filename}…"):
                kbs.append(build_kb_for_doc(doc))
            progress.progress((i + 1) / len(docs))
        st.session_state.knowledge_base = merge_kbs(kbs)

    kb = st.session_state.knowledge_base
    if kb:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-label">Themes</div>', unsafe_allow_html=True)
            for t in kb.themes[:8]:
                st.markdown(f"- {t}")
            st.markdown('<div class="section-label">Methods</div>', unsafe_allow_html=True)
            for m in kb.methods[:6]:
                st.markdown(f"- {m}")
        with col2:
            st.markdown('<div class="section-label">Findings</div>', unsafe_allow_html=True)
            for f in kb.findings[:6]:
                st.markdown(f"- {f}")
            st.markdown('<div class="section-label">Sources</div>', unsafe_allow_html=True)
            for c in kb.citations:
                st.markdown(f"- [{c.key}] {c.title[:60]}…")

        nav_buttons(back_step=2, next_step=4, next_label="Configure Paper →")
    else:
        nav_buttons(back_step=2)

def step_configure():
    st.markdown("## 4 · Configure Paper")
    cfg = st.session_state.paper_config

    cfg.title = st.text_input("Paper title", value=cfg.title)
    authors_raw = st.text_area("Authors (one per line)", value="\n".join(cfg.authors), height=80)
    cfg.authors = [a.strip() for a in authors_raw.split("\n") if a.strip()]
    affiliations_raw = st.text_area("Affiliations (one per line)", value="\n".join(cfg.affiliations), height=60)
    cfg.affiliations = [a.strip() for a in affiliations_raw.split("\n") if a.strip()]
    cfg.corresponding_author = st.text_input("Corresponding author", value=cfg.corresponding_author)
    cfg.email = st.text_input("Contact email", value=cfg.email)
    keywords_raw = st.text_input("Keywords (comma-separated)", value=", ".join(cfg.keywords))
    cfg.keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()]

    col1, col2, col3 = st.columns(3)
    with col1:
        cfg.template = st.selectbox("Template", ["ieee", "generic", "acm (stub)", "springer (stub)"], index=["ieee", "generic", "acm (stub)", "springer (stub)"].index(cfg.template) if cfg.template in ["ieee", "generic", "acm (stub)", "springer (stub)"] else 0)
    with col2:
        cfg.layout = st.selectbox("Column layout", ["double", "single"], index=0 if cfg.layout == "double" else 1)
    with col3:
        cfg.paper_type = st.selectbox("Paper type", ["journal", "conference"], index=0 if cfg.paper_type == "journal" else 1)

    cfg.target_words = st.slider("Target word count", 2000, 12000, cfg.target_words, step=500)
    st.session_state.paper_config = cfg

    nav_buttons(back_step=3, next_step=5, next_label="Generate Outline →")

def step_outline():
    st.markdown("## 5 · Generate Outline")
    if not st.session_state.knowledge_base:
        st.warning("Knowledge base missing. Go back.")
        nav_buttons(back_step=4)
        return

    if st.button("Generate Outline", type="primary"):
        with st.spinner("Generating outline…"):
            outline = generate_outline(st.session_state.knowledge_base, st.session_state.paper_config)
        st.session_state.outline = outline

    outline = st.session_state.outline
    if outline:
        for sec, bullets in outline.items():
            with st.expander(sec.replace("_", " ").title()):
                for b in (bullets if isinstance(bullets, list) else [str(bullets)]):
                    st.markdown(f"- {b}")
        nav_buttons(back_step=4, next_step=6, next_label="Generate Sections →")
    else:
        nav_buttons(back_step=4)

def step_generate():
    st.markdown("## 6 · Generate Sections")
    outline = st.session_state.outline
    kb = st.session_state.knowledge_base
    docs = st.session_state.extracted_documents
    cfg = st.session_state.paper_config

    if not outline or not kb:
        st.warning("Outline or knowledge base missing.")
        nav_buttons(back_step=5)
        return

    section_fns = [
        ("abstract", lambda: generate_abstract(kb, outline, cfg)),
        ("introduction", lambda: generate_introduction(kb, outline, cfg)),
        ("literature_review", lambda: generate_literature_review(kb, outline)),
        ("methodology", lambda: generate_methodology(kb, outline, docs)),
        ("results", lambda: generate_results(kb, outline, docs)),
        ("discussion", lambda: generate_discussion(kb, outline)),
        ("conclusion", lambda: generate_conclusion(kb, outline, cfg)),
    ]

    if st.button("Generate All Sections", type="primary"):
        generated_raw: dict[str, tuple[str, list[str]]] = {}
        progress = st.progress(0)
        for i, (name, fn) in enumerate(section_fns):
            with st.spinner(f"Writing {name.replace('_', ' ').title()}…"):
                text, cits = fn()
                generated_raw[name] = (text, cits)
            progress.progress((i + 1) / len(section_fns))

        st.session_state.generated_sections = {k: v[0] for k, v in generated_raw.items()}
        st.session_state.citation_map = build_citation_map(generated_raw)

    if st.session_state.generated_sections:
        for sec, text in st.session_state.generated_sections.items():
            label = sec.replace("_", " ").title()
            with st.expander(f"✏️ {label}"):
                edited = st.text_area(f"Edit {label}", value=text, height=250, key=f"edit_{sec}")
                st.session_state.generated_sections[sec] = edited

        nav_buttons(back_step=5, next_step=7, next_label="Citations →")
    else:
        nav_buttons(back_step=5)

def step_citations():
    st.markdown("## 7 · Citations")
    kb = st.session_state.knowledge_base
    if not kb:
        st.warning("Knowledge base missing.")
        nav_buttons(back_step=6)
        return

    bib = dedup_citations(kb.citations)
    st.session_state.bibliography = bib

    st.markdown(f"**{len(bib)} unique sources found.**")
    for c in bib:
        missing = []
        if not c.doi:
            missing.append("DOI")
        if not c.year:
            missing.append("year")
        badge = " ⚠️ Missing: " + ", ".join(missing) if missing else " ✅"
        with st.expander(f"[{c.key}] {c.title[:60]}{badge}"):
            c.title = st.text_input("Title", value=c.title, key=f"cit_title_{c.key}")
            c.year = st.text_input("Year", value=c.year, key=f"cit_year_{c.key}")
            c.journal = st.text_input("Journal", value=c.journal, key=f"cit_journal_{c.key}")
            c.doi = st.text_input("DOI", value=c.doi, key=f"cit_doi_{c.key}")
            authors_raw = st.text_input("Authors (semicolon-separated)", value="; ".join(c.authors), key=f"cit_auth_{c.key}")
            c.authors = [a.strip() for a in authors_raw.split(";") if a.strip()]

    nav_buttons(back_step=6, next_step=8, next_label="Evidence Coverage →")

def step_evidence():
    st.markdown("## 8 · Evidence Coverage")
    sections = st.session_state.generated_sections
    cm = st.session_state.citation_map
    if not sections:
        st.warning("No generated sections found.")
        nav_buttons(back_step=7)
        return

    scores = compute_evidence(sections, cm)
    st.session_state.evidence_scores = scores

    if scores:
        overall = sum(s.coverage_pct for s in scores) / len(scores)
        st.metric("Average coverage", f"{overall:.1f}%")

        for sec in {s.section for s in scores}:
            sec_scores = [s for s in scores if s.section == sec]
            avg = sum(s.coverage_pct for s in sec_scores) / len(sec_scores) if sec_scores else 0
            cls = "cov-green" if avg >= 70 else ("cov-amber" if avg >= 30 else "cov-red")
            st.markdown(
                f'<div class="card"><b>{sec.replace("_"," ").title()}</b> — '
                f'<span class="{cls}">{avg:.0f}% coverage</span> across {len(sec_scores)} paragraph(s)</div>',
                unsafe_allow_html=True,
            )
            unsupported = [s for s in sec_scores if s.unsupported]
            if unsupported:
                st.warning(f"⚠️ {len(unsupported)} unsupported paragraph(s) in {sec} — add citations before export.")
    else:
        st.info("No paragraphs to score.")

    nav_buttons(back_step=7, next_step=9, next_label="Preview LaTeX →")

def step_preview():
    st.markdown("## 9 · Preview LaTeX")
    cfg = st.session_state.paper_config
    sections = st.session_state.generated_sections
    bibliography = st.session_state.bibliography

    if not sections:
        st.warning("No sections generated.")
        nav_buttons(back_step=8)
        return

    latex_files = build_latex_files(cfg, sections, bibliography)
    st.session_state.latex_files = latex_files

    for fname, content in latex_files.items():
        with st.expander(f"📄 {fname}"):
            st.code(content, language="latex" if fname.endswith(".tex") or fname.endswith(".bib") else "markdown")

    nav_buttons(back_step=8, next_step=10, next_label="Export →")

def step_export():
    st.markdown("## 10 · Export")
    cfg = st.session_state.paper_config
    sections = st.session_state.generated_sections
    scores = st.session_state.evidence_scores
    bibliography = st.session_state.bibliography
    latex_files = st.session_state.latex_files
    docs = st.session_state.extracted_documents

    if not latex_files:
        st.warning("No LaTeX files. Go back to Preview.")
        nav_buttons(back_step=9)
        return

    total_words = sum(len(t.split()) for t in sections.values())
    avg_cov = sum(s.coverage_pct for s in scores) / len(scores) if scores else 0
    unsupported_count = sum(1 for s in scores if s.unsupported)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sections", len(sections))
    col2.metric("Est. words", f"{total_words:,}")
    col3.metric("References", len(bibliography))
    col4.metric("Avg. coverage", f"{avg_cov:.0f}%")

    if unsupported_count:
        st.warning(f"⚠️ {unsupported_count} paragraph(s) have no source citations. Review before submission.")

    try:
        zip_bytes = build_zip(latex_files, docs)
        st.download_button(
            label="⬇️ Download project.zip",
            data=zip_bytes,
            file_name="research_paper_project.zip",
            mime="application/zip",
            type="primary",
        )
    except Exception as e:
        st.error(f"ZIP export failed: {e}")

    nav_buttons(back_step=9)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="AI Research Paper Generator", layout="wide", page_icon="📄")
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()

    st.markdown("# 📄 AI Research Paper Generator")
    render_step_bar(st.session_state.current_step)

    step = st.session_state.current_step
    step_fns = {
        1: step_upload,
        2: step_extract,
        3: step_kb,
        4: step_configure,
        5: step_outline,
        6: step_generate,
        7: step_citations,
        8: step_evidence,
        9: step_preview,
        10: step_export,
    }
    step_fns.get(step, step_upload)()

if __name__ == "__main__":
    main()


