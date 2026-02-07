# 🎵 Chord Detector

Drag-and-drop chord progression analyser with real-time audio playback, interactive piano, circle of fifths, guitar voicings, harmonic tension graphing, and more.

## Features

- **Chord detection** — CQT chromagram + template matching across 14 chord qualities × 12 roots
- **Audio player** — synced playhead with waveform visualisation
- **Interactive piano** — highlights chord tones in real time as audio plays
- **Roman numeral analysis** — every chord labelled relative to the detected key
- **Famous pattern matching** — recognises 14 well-known progressions (Axis of Awesome, Andalusian Cadence, Jazz ii-V-I, etc.)
- **Circle of fifths** — SVG visualisation of the harmonic journey
- **Chord transition matrix** — heatmap of which chords move to which
- **Harmonic tension graph** — consonance/dissonance plotted over time
- **Key change detection** — windowed tonal analysis detects modulations
- **Spectrogram + chromagram** — toggle between frequency views
- **Guitar chord diagrams** — SVG fingering charts for 70+ voicings
- **Scale suggestions** — click any chord to see which scales work for soloing
- **Export** — MIDI, formatted text, or JSON

---

## Deploy to Railway (Recommended)

Railway is the fastest way to get this live. Total time: ~5 minutes.

### Prerequisites

- A [GitHub](https://github.com) account
- A [Railway](https://railway.app) account (sign up with GitHub)

### Steps

**1. Create a GitHub repo**

```bash
cd chord-detector-web
git init
git add .
git commit -m "Initial commit"
```

Go to [github.com/new](https://github.com/new), create a new repo (public or private), then:

```bash
git remote add origin https://github.com/YOUR_USERNAME/chord-detector.git
git branch -M main
git push -u origin main
```

**2. Deploy on Railway**

1. Go to [railway.app/new](https://railway.app/new)
2. Click **"Deploy from GitHub Repo"**
3. Select your `chord-detector` repo
4. Railway auto-detects the Dockerfile and starts building
5. Wait ~2-3 minutes for the build to complete
6. Click **"Settings" → "Networking" → "Generate Domain"** to get your public URL

That's it. Your app is live at `https://chord-detector-XXXX.up.railway.app`.

**3. (Optional) Custom domain**

In Railway Settings → Networking → Custom Domain, add your domain and point a CNAME record to the Railway URL.

### Costs

- **Free tier**: $5/month credit, no credit card required. Enough for personal use.
- **Hobby tier**: $5/month flat. No cold starts, more CPU time.
- Audio analysis is CPU-heavy — a single 5-minute song takes ~10-15 seconds to process.

### Auto-deploy

Every `git push` to `main` triggers a new deployment automatically. Zero-downtime deploys.

---

## Alternative Hosting

### Render

```bash
# Push to GitHub first (same as above), then:
# 1. Go to render.com → New → Web Service
# 2. Connect your repo
# 3. Render auto-detects the Dockerfile
# 4. Set start command (if needed): gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 300 app:app
```

Free tier has cold starts (~30-60s). Paid tier ($7/month) stays warm.

### Fly.io

```bash
# Install flyctl: https://fly.io/docs/getting-started/installing-flyctl/
cd chord-detector-web
fly launch     # Follow prompts, select region
fly deploy
```

### Docker (local or any VPS)

```bash
docker compose up --build -d
# App runs at http://localhost:5000
```

For a VPS (DigitalOcean, Hetzner, etc.), just SSH in and run the above. Add Caddy or nginx for HTTPS.

### DigitalOcean / Hetzner VPS

```bash
# SSH into your server
git clone https://github.com/YOUR_USERNAME/chord-detector.git
cd chord-detector
docker compose up --build -d

# Add Caddy for auto-HTTPS (optional)
sudo apt install caddy
echo "chords.yourdomain.com { reverse_proxy localhost:5000 }" | sudo tee /etc/caddy/Caddyfile
sudo systemctl restart caddy
```

---

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Also need ffmpeg for audio format support
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg libsndfile1

# Run
python app.py
# → http://localhost:5000
```

---

## Project Structure

```
chord-detector-web/
├── app.py                 # Flask backend + analysis engine
├── templates/
│   └── index.html         # Complete frontend (single file, no build step)
├── Dockerfile             # Production container
├── docker-compose.yml     # One-command local deployment
├── railway.toml           # Railway deployment config
├── requirements.txt       # Python dependencies
├── .dockerignore
└── .gitignore
```

## Tech Stack

- **Backend**: Python, Flask, librosa, NumPy, SciPy, midiutil
- **Frontend**: Vanilla HTML/CSS/JS, Canvas API, SVG (no build tools)
- **Audio**: CQT chromagram, HPSS harmonic separation, beat tracking
- **Theory**: Krumhansl-Schmuckler key detection, Helmholtz tension scoring
