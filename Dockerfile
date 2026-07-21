FROM python:3.12-slim

WORKDIR /app

# Najpierw sama lista zależności: dopóki requirements.txt się nie zmienia,
# ta warstwa idzie z cache i build po zmianie kodu jest szybki.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Dopiero teraz kod (bez plików z .dockerignore — m.in. bez .env).
COPY . .

# Logi widoczne od razu w konsoli Rendera (bez buforowania).
ENV PYTHONUNBUFFERED=1

# Forma shell celowo: Render wstrzykuje zmienną PORT, lokalnie fallback 8000.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
