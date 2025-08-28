# ArchAI

A proof-of-concept application that uses AI to detect and analyze floorplan elements from uploaded images. The project consists of a Next.js frontend for user interaction and a Python Flask API backend for image processing and floorplan analysis. Users can upload floorplans, view them in an interactive 3D scene, take snapshots, and generate photorealistic renders with AI-powered staging.

## Architecture

This project follows a microservices architecture with three main components:

### Frontend (Next.js)

- **Framework**: Next.js 15.4.2 with App Router
- **UI**: React 19 with Tailwind CSS and Radix UI components
- **3D Visualization**: Three.js with React Three Fiber for 3D floorplan rendering
- **State Management**: Zustand for global state, React Query for server state
- **API Routes**: Next.js API routes for AI rendering and staging services
- **Port**: Runs on `http://localhost:3000`

### Backend (Python Flask API)

- **Framework**: Flask 3.0.3
- **AI/ML**: PyTorch, OpenCV, scikit-image for image processing
- **Notebook Processing**: Jupyter notebook execution for floorplan analysis
- **Containerization**: Docker for consistent deployment
- **Port**: Runs on `http://localhost:5050`

### Backend (ComfyUI/Stable Diffusion API)

- **Framework**: ComfyUI
- **AI/ML**: PyTorch, OpenCV, scikit-image for image processing
- **Notebook Processing**: Jupyter notebook execution for floorplan analysis
- **Containerization**: Docker for consistent deployment
- **Port**: Runs on `http://localhost:5050`

### API Endpoints

#### Next.js API Routes (`/app/api/`)

- `POST /api/gen-render` - Generate photorealistic renders using external AI service
- `POST /api/gen-staging` - Create staged renders with furniture and decor using OpenAI

#### Python Flask API (`/backend/`)

- `POST /vertx` - Process floorplan image and return detected elements

## Features

- **Image Upload**: Drag-and-drop interface for floorplan images
- **AI Analysis**: Automatic detection of:
  - Doors
  - Windows
  - Walls
  - Floor areas
  - Canvas boundaries
- **3D Visualization**: Interactive 3D rendering of detected elements
- **Real-time Processing**: Immediate analysis and visualization

## Prerequisites

- Node.js 18+ and npm/pnpm/bun
- Python 3.11+
- Docker (for backend)

## Getting Started

**Install frontend dependencies**:
```bash
pnpm install
```


### Option 1: One-command startup (Recommended)

**Start the app**:
```bash
npm start
# or 
pnpm start
```
This will automatically start:

- Next.js frontend on http://localhost:3000
- Python Flask API on http://localhost:5050
- All required Docker services in the background

⚠️ Make sure Docker Desktop is running before executing this command.


### Option 2: Manual startup

1. **Start the frontend**:

   ```bash
   npm run dev:next
   ```

2. **Start the backend** (in a separate terminal):
   ```bash
   npm run dev:backend
   ```

## Development

### Frontend Development

- The main application is in `app/(floorplan)/page.tsx`
- Components are located in `app/(floorplan)/_components/`
- Custom hooks are in the `hooks/` directory
- TypeScript types are defined in `types/`

### Backend Development

- Main API server: `backend/index.py`
- Floorplan detection logic: `backend/flooplan_detector.ipynb`
- Utility functions: `backend/utils/`
- ML models: `backend/model/`

## Technologies Used

### Frontend

- Next.js 15.4.2
- React 19
- TypeScript
- Tailwind CSS
- Radix UI
- Three.js / React Three Fiber
- Zustand
- React Query
- React Hook Form
- Zod validation

### Backend

- Flask 3.0.3
- PyTorch 2.7.0
- OpenCV 4.11.0
- scikit-image 0.25.2
- Jupyter notebook processing
- Docker containerization

## Project Structure
```
├── app/ # Next.js app directory
│ ├── (floorplan)/ # Main floorplan application
│ │ ├── components/ # React components
│ │ └── page.tsx # Main page
│ ├── api/ # Next.js API routes
│ ├── globals.css # Global styles
│ └── layout.tsx # Root layout
├── backend/ # Python Flask API
│ ├── index.py # Main Flask application
│ ├── flooplan_detector.ipynb # AI analysis notebook
│ ├── requirements.txt # Python dependencies
│ ├── dockerfile # Docker configuration
│ ├── model/ # ML models
│ └── utils/ # Utility functions
├── components/ # Shared React components
├── hooks/ # Custom React hooks
├── lib/ # Utility libraries
├── types/ # TypeScript type definitions
└── package.json # Node.js dependencies
```

## Floorplan-Detection Logic Notebook

* **Location:** `backend/floorplan_detector.ipynb`
* This notebook documents and explores the complete floorplan detection pipeline.
* It is directly integrated with the app—any modifications here will affect detection results.
* Check it out to understand how the `floorplan_detector` works.

> **Debug Visualization:**  
If you want to see the different maps that run simultaneously in debug mode, you can use the following keyboard shortcuts while the app is running:
- Press **1** to view the **Depth Map**
- Press **2** to view the **Line-Segment Map**
- Press **3** to view the **Color-Segment Map**
- Press **0** to return to the regular scene

This feature is useful for inspecting the intermediate outputs of the floorplan detection pipeline during development or debugging.


