# HR Recruiter Assistant

This project is an AI-powered recruiter assistant that helps you build a list of people to interview by searching the internet (especially LinkedIn) and gathering relevant information about potential candidates.

## Features
- Uses OpenAI's GPT-4o model and LangChain for intelligent search and information extraction
- Searches for people based on your criteria and retrieves:
  - Name
  - Location
  - Job
  - Company
  - Key strengths (concise bullet points)
  - LinkedIn profile link
- Iteratively refines search queries to find the most relevant candidates
- Designed to keep responses concise and focused

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies

## Setup
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd linkedin
   ```
2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up environment variables:**
   - Create a `.env` file in the project root with your API keys (e.g., OpenAI, Tavily, etc.) as required by the dependencies.

## Usage
Run the assistant with:
```bash
python linkedin.py
```

You will interact with the recruiter assistant, who will search for candidates and build a list based on your input.

## Notes
- The `.env`, `.venv`, and other sensitive files are gitignored for security.
- This project uses the [Davia](https://pypi.org/project/davia/) framework for the app interface.