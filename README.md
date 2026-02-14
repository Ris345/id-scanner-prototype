# ID Scanner

Scan government-issued IDs (driver's licenses, etc.) using your camera or photo gallery. Extracts name, date of birth, address, ID number, and state using AWS Textract.

## Architecture

- **Frontend:** Expo (React Native) — runs on iOS, Android, and web
- **Backend:** Node.js/Express server that handles image processing and AWS Textract calls
- **OCR:** AWS Textract AnalyzeID

In development, the app hits a local backend (`localhost:3001`). In production, it points to the deployed backend on Render.

## Development Setup

### 1. Backend

```bash
cd backend
npm install
```

Create a `.env` file in the project root with **your own** AWS credentials:

```
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1
```

Start the backend:

```bash
npm run dev
```

The backend runs on `http://localhost:3001`.

### 2. Frontend

```bash
npm install
npx expo start
```

Press `w` to open in the browser, or scan the QR code with Expo Go for mobile.

**Important:** In dev mode, the app connects to `localhost:3001`. You must run your own backend with your own AWS credentials. The production Render backend is not for development use.

## Production

- **Live Demo:** [https://69910ad24cfd88129a76a10d--incomparable-donut-edf69d.netlify.app/](https://69910ad24cfd88129a76a10d--incomparable-donut-edf69d.netlify.app/)
- **Backend** is deployed on [Render](https://render.com) at `https://id-scanner-prototype.onrender.com`
- **Frontend** can be redeployed as a static web export to Netlify/Vercel:

```bash
npx expo export --platform web
```

This outputs to `dist/` which can be deployed to any static hosting provider.
