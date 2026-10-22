# 📄 AI Research Paper Generator

An AI-powered tool that helps researchers generate structured academic papers from uploaded PDF literature using Google Gemini AI.

Upload your reference papers → AI extracts knowledge → Generates a complete research paper with citations → Export as LaTeX.

## ✨ Features

- **PDF Extraction** — Upload multiple research papers; extracts text, sections, figures, tables, and metadata using PyMuPDF
- **Knowledge Base Synthesis** — AI analyzes papers to extract themes, methods, datasets, findings, and limitations
- **Smart Outline Generation** — Generates a structured paper outline based on synthesized knowledge
- **Section-by-Section Generation** — AI writes each section with proper citations grounded in source documents
- **Citation Mapping** — Automated citation linking between generated content and source papers
- **Evidence Scoring** — Validates every paragraph against source documents for citation coverage
- **LaTeX Export** — Professional LaTeX output supporting IEEE, ACM, and APA templates
- **ZIP Download** — One-click export of all generated files (LaTeX, bibliography, figures)

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| Python | Core language |
| Streamlit | Web UI framework |
| Google Gemini 2.0 | AI text generation and analysis |
| PyMuPDF (Fitz) | PDF parsing and extraction |
| Pydantic | Data validation and models |
| Jinja2 | LaTeX template rendering |

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Google Gemini API key ([Get one here](https://aistudio.google.com/apikey))

### Installation

```bash
# Clone the repository
git clone https://github.com/nazifa-ai/AI-Research-Paper-Generator.git
cd AI-Research-Paper-Generator

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.streamlit/secrets.toml` file:

```toml
GEMINI_API_KEY = "your-gemini-api-key-here"
```

Or set it as an environment variable:

```bash
export GEMINI_API_KEY="your-gemini-api-key-here"
```

### Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📋 How It Works

```
Step 1: Upload PDFs        → Upload your reference research papers
Step 2: Extract            → AI extracts sections, metadata, figures
Step 3: Knowledge Base     → Synthesizes themes, methods, findings
Step 4: Configure          → Set title, authors, template format
Step 5: Generate Outline   → AI creates structured paper outline
Step 6: Generate Sections  → AI writes each section with citations
Step 7: Citation Mapping   → Links citations to source documents
Step 8: Evidence Scoring   → Validates claims against sources
Step 9: Preview            → Review the generated paper
Step 10: Export            → Download as LaTeX + ZIP
```

## 📂 Project Structure

```
AI-Research-Paper-Generator/
├── app.py               # Main application (Streamlit)
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── .gitignore           # Git ignore rules
└── README.md            # Project documentation
```

## 🌐 Live Demo

> Deployed on [Streamlit Cloud](https://share.streamlit.io) — [View Live Demo](#)

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👩‍💻 Author

**Nazifa Nadkar**
- GitHub: [@nazifa-ai](https://github.com/nazifa-ai)
- LinkedIn: [Nazifa Nadkar](https://linkedin.com/in/nazifa-nadkar)
