# Paper Drum — 6 Pads (Acoustic)

A browser-based AR-style drum pad. Print the sheet, point your phone camera, tap the circles, hear drums.
Built with **MediaPipe Hands** + **Web Audio API**. No server required.

## Live Demo (GitHub Pages)
This repo is structured to deploy on **GitHub Pages**.

### Quick Deploy
1. Create a **new GitHub repository** (public or private).
2. **Upload** these files to the repo (root): `index.html`, `style.css`, `app.js`, `assets/`, `sounds/`, `.nojekyll`, `README.md`.
3. Go to **Settings → Pages**:
   - **Source**: choose your default branch (e.g., `main`).
   - **Root**: `/ (root)`.
   - Click **Save**.
4. Wait a minute; GitHub shows your site URL (HTTPS). Open it on your **phone**.
   - The app requests **camera access**; allow it.

> Tip: If you see a 404 right after enabling Pages, wait 1–2 minutes and refresh. GitHub needs a moment to build the static site.

## How to Use
1. Print `assets/printable_paper_drum_6pad.pdf` (US Letter).
2. Open the site on your **phone** → tap **Start Camera** → allow access.
3. Keep the phone roughly **square** to the paper (this MVP uses simplified calibration).
4. Tap pads with your **index fingertip**.

## Project Notes
- **MediaPipe Hands** is loaded via CDN.
- **Acoustic drum samples** are bundled as WAV files for low latency.
- Hit logic uses fingertip velocity + cooldown to avoid double triggers.
- For best results: **bright, even lighting**.

## Next Steps (optional)
- Add 4-corner marker detection and a **perspective homography** to handle tilt.
- Sensitivity slider (velocity threshold).
- Multi-finger concurrency and custom kits.

## Local Preview (optional)
Serve the folder with a small local server (to avoid camera restrictions and CORS issues):

```bash
# Python 3
python -m http.server 8000
# then visit http://localhost:8000
```

## License
You may use and modify this project for personal or educational purposes.
