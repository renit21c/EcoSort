# EcoSort AI — garbage classification web app

A portfolio-ready Next.js + TypeScript app using the supplied TensorFlow.js GraphModel. Includes client-side inference, recycling guidance, Supabase email/password auth, per-user history, and analytics. It works in local demo mode without Supabase; authentication and cloud history activate after environment setup.

## Model details verified from the supplied archive

- Format: TensorFlow.js **GraphModel** (`tf.loadGraphModel`), not LayersModel.
- Input: float32 tensor `[1, 224, 224, 3]`, RGB pixels `0–255`.
- Output: `[1, 12]` softmax probabilities.
- The model includes its own `Rescaling(1/255)` layer. **Do not normalize the input again.**
- Output order: `battery, biological, brown-glass, cardboard, clothes, green-glass, metal, paper, plastic, shoes, trash, white-glass`.
- README in the source archive contains a conflicting 150×150 example; the exported model metadata says 224×224, which this app uses.
- Reported held-out test accuracy: 85.0%; this is the model author's evaluation, not a guarantee on real-world photos.

## Live camera scanning

The classifier page can use the device camera to detect visible objects. Select a detection box to capture that object, then run the waste classifier on the cropped image. Camera access requires a secure context (HTTPS or localhost) and browser permission. The object detector is loaded on first camera use; inference and image classification run in the browser.

## Run locally

1. Install Node.js 20+.
2. From this directory run:
   ```bash
   npm install
   cp .env.example .env.local
   npm run dev
   ```
3. Open http://localhost:3000. If Supabase env vars are not configured, you can use the classifier and local-only history in demo mode.

## Enable accounts, cloud history and analytics

1. Create a Supabase project.
2. In Supabase SQL Editor, run `supabase/schema.sql`.
3. Copy Project URL and anon/public key into `.env.local` using the names in `.env.example`.
4. Restart `npm run dev`.
5. For email confirmation, set the Supabase Auth Site URL to `http://localhost:3000`; add your deployed URL before production deployment.

The app only inserts classification metadata (label, confidence, probabilities and timestamp), not the uploaded image. Row Level Security restricts history to the authenticated owner. Never put a Supabase service-role key in a `NEXT_PUBLIC_` variable or browser code.

## Deploy

Push to GitHub, import the repository into Vercel, add the two `NEXT_PUBLIC_SUPABASE_*` variables, run the SQL schema in Supabase, and set Supabase Auth's Site URL / redirect URLs to the production domain. Vercel builds with `npm run build`.

## Important production notes

- This is a starter product, not a certified waste-safety tool. Show the guidance as general advice and check local waste rules.
- Predictions can be wrong, especially for visually similar items and hazardous waste. Encourage users to verify battery/e-waste and chemical disposal independently.
- Inference runs in the browser. The first classification downloads roughly 22 MB of model files and can take a moment on slower devices.
- The model archive includes its own dataset/model README; retain that attribution when redistributing the model and review the dataset's license/terms before public commercial use.
