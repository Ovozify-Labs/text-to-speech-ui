# Text To Speech UI Application

A modern Text to Speech application with a Next.js frontend and FastAPI backend.

## Technology Stack

- **Frontend**: Next.js (React framework)
- **Backend**: FastAPI (Python)
- **UI Components**: Lucide React icons

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.10
- Node.js 24.11.0
- npm package manager

## Installation & Setup

### Backend Setup

1. Navigate to the project root directory

2. Create a Python virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

4. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Start the FastAPI backend server:
   ```bash
   uvicorn main:app --port 8000 --host 0.0.0.0
   ```

   The backend API will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install npm dependencies:
   ```bash
   npm install
   ```

3. Install Lucide React icons:
   ```bash
   npm install lucide-react
   ```

4. Start the development server:
   ```bash
   npm run dev
   ```

   The frontend application will be available at `http://localhost:3000`

## Running the Application

1. Ensure the backend server is running on port 8000
2. Ensure the frontend development server is running on port 3000
3. Open your browser and navigate to `http://localhost:3000`

## Project Structure

```
.
├── frontend/           # Next.js frontend application
│   ├── components/     # React components
│   ├── pages/          # Next.js pages
│   └── package.json    # Frontend dependencies
├── main.py              # FastAPI backend application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Troubleshooting

- **Backend won't start**: Ensure your virtual environment is activated and all dependencies are installed
- **Frontend won't start**: Make sure you're in the `frontend` directory and have run `npm install`
- **Port conflicts**: If ports 8000 or 3000 are already in use, modify the port numbers in the startup commands

## Contributing

Feel free to submit issues and enhancement requests!