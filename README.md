1. Run `uv sync` command in a terminal into the project directory to setup the necessary libraries.

2. Activate the created virtual environment.

```
source .venv/bin/activate
# .venv\Scripts\activate (on Windows)
```

3. Set your API key. Copy `.env.example` to `.env` and fill in your key:

```
cp .env.example .env
# then edit .env and set OPENAI_API_KEY=sk-...
```

4. Run the `client.py` file.

```
uv run client.py
```
