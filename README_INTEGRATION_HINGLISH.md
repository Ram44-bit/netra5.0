# NETRA AI — UI Refresh + Frontend/Backend ZIP

Bhai, is build mein supplied HTML layout ko retain karke dark teal, accessible, mobile-first styling di gayi hai. Existing browser COCO-SSD object detection, camera, image upload, text-to-speech, speech input, OpenAI chat endpoint, and Leaflet/OpenStreetMap map features included hain.

## Project folders

- `netra-ai-frontend/` — static HTML/CSS/JS site for Render Static Site
- `netra-ai-backend/` — FastAPI backend for OpenAI chat and optional Gemini image description

## Frontend deploy (Render Static Site)

1. `netra-ai-frontend` folder ki files apni frontend repository ke root mein upload/replace karo.
2. Render Static Site settings mein publish directory `.` (ya repo root, if frontend files are at root) set karo.
3. HTTPS URL khol kar camera permission allow karo. First load par TensorFlow.js/COCO-SSD model internet se download hota hai; pehli baar isliye thoda wait ho sakta hai.

## Backend deploy (Render Web Service)

Root directory: `netra-ai-backend`

- Build/install: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

Environment variables:

- `OPENAI_API_KEY` — required for OpenAI chat
- `OPENAI_MODEL` — optional; default `gpt-4.1-mini`
- `FRONTEND_ORIGIN` — exact deployed frontend URL, e.g. `https://your-site.onrender.com`
- `GEMINI_API_KEY` — needed for Detailed AI Scan and `/api/describe` image analysis
- `GEMINI_MODEL` — optional, default `gemini-2.5-flash`

After deploying the backend, make sure the frontend's `API_BASE` in `app.js` points to the real backend service URL. It currently defaults to the existing NETRA backend URL and can be changed at the `API_BASE` constant near the chat code.

## Test checklist

1. Open backend `/health` and confirm `openai_chat_configured: true` if OpenAI is configured.
2. Open the frontend over HTTPS and allow camera access.
3. Wait for `AI model ready`; then start camera or upload a photo.
4. Ask a typed question in Ask NETRA. Chat should answer only when backend URL, CORS and `OPENAI_API_KEY` are correctly set.
5. Press `Quick detect` for fast common-object detection. For tiny items or a visible facial expression, press `Detailed AI scan`; it needs `GEMINI_API_KEY`, internet, and may take a few seconds.
6. Press `Read results aloud` to replay the latest result. Speech clarity also depends on the device/browser having a good Hindi voice installed; this build selects a Hindi voice when available and slows speech slightly.
7. Test map location only after pressing `My location`, then search a destination or nearby services.

## Important safety & privacy notes

- Browser object detection runs locally in the browser; loading the model requires internet on first use.
- Chat sends the typed question and detected-object labels to your configured backend. Do not put API keys in frontend files.
- Camera images are not automatically sent to OpenAI chat. Pressing `Detailed AI scan` intentionally sends the current still frame/photo to the configured Gemini vision endpoint for analysis; it is processed for that request and not written to disk by this backend.
- Map results use public OpenStreetMap-related services and may be incomplete or temporarily unavailable.
- NETRA is a prototype, not a certified mobility aid. Do not rely on it to cross roads or determine whether a route is safe.

## Accuracy notes

COCO-SSD does not have classes for pen, pencil, USB drive or charger, so Quick detect cannot reliably name those items. Detailed AI Scan uses a vision-language model instead and can still miss or misidentify small/occluded items. Better lighting, holding the item closer, and keeping it in focus helps. Facial expression is only a tentative visible-expression estimate, not a reading of the person's actual feelings.
