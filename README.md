# NETRA AI — Frontend Prototype

A responsive, static HTML/CSS/JavaScript prototype that can be deployed to Render Static Sites. It includes camera preview, photo upload, in-browser common-object detection with TensorFlow.js + COCO-SSD, speech output, browser-supported speech input, and an OpenStreetMap-based map module.

## Files
- `index.html` — page structure and external browser AI libraries
- `style.css` — responsive styling
- `app.js` — camera, object detection, voice controls, demo Q&A, and map interactions
- `render.yaml` — optional Render Blueprint configuration

## Run locally
For best results, use a local web server rather than opening `index.html` directly:
1. Open this folder in VS Code.
2. In a terminal, run `python -m http.server 8000`
3. Open `http://localhost:8000`

Camera access usually works on `localhost` and on secure HTTPS websites.

## Deploy on Render
1. Upload these files to a GitHub repository.
2. In Render, choose **New + → Static Site** and connect that repository.
3. Set **Build Command** to blank (no build step is needed).
4. Set **Publish Directory** to `.` (the folder containing `index.html`).
5. Click **Deploy**. Open the Render URL over HTTPS and allow camera permission.

Alternatively, use the included `render.yaml` Blueprint configuration if deploying via Render Blueprints.

## What works in this frontend-only demo
- Camera preview (browser permission required)
- Upload a photo
- Real browser-side detection of common objects, when the TensorFlow.js libraries and model download successfully
- Detection labels and confidence scores
- Read detection results aloud with browser text-to-speech
- Speech input on browsers that support Web Speech API
- Basic demo answers
- Current location after explicit browser permission
- Destination search with map marker
- Nearby mapped pharmacies, hospitals, cafes, restaurants, bus stops, and public toilets
- External Google Maps walking-directions link when both start and destination are known

## What is not connected yet
- Gemini/other generative AI answers for arbitrary questions
- Backend, accounts, persistent storage, complaint/report submission, or cloud analytics

## Map service notes
- Map tiles use OpenStreetMap; destination search uses Nominatim; nearby places use the public Overpass API. These public services can be rate-limited or temporarily unavailable.
- Location is requested only when the user presses **My location**. The map starts at a general India view, not a guessed user location.
- Walking directions open Google Maps in a new tab; this prototype does not calculate or verify a safe route itself.

## Important limitations
- The first model load needs an internet connection and may take a little time.
- Object detection can miss objects or mislabel them. This is not a certified assistive/navigation or safety device. Do not rely on it to cross roads or avoid hazards.
- Camera and microphone permissions are controlled by the browser.
- Web Speech API support varies by browser and device. The microphone feature may not be available in every browser.
- This prototype uses third-party CDN scripts for TensorFlow.js and COCO-SSD. If the CDN is blocked, AI detection will not load.
