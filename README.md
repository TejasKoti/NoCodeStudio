# 🧠 NoCodeStudio : Visual Model Builder & Trainer

A full-stack, end-to-end **no-code deep learning platform** that allows
users to visually design neural networks, automatically generate PyTorch
code, train models with built-in datasets, and run inference which is all
inside a polished, modern web UI where users can : 

- Drag & drop layers to build neural network graphs
- Import or export PyTorch model code
- Train models using default or custom datasets
- Upload custom model code or custom training logic
- Manage projects, folders, and templates
- Run inference with JSON inputs
- Use a fully isolated Python engine handling training/inference

NoCodeStudio functions as a **lightweight no-code ML IDE**, powered by a
React frontend, TypeScript API, and Python FastAPI training engine.

---

## 🖼 Screenshots

- **Landing Page**  
  ![Landing](NoCodeStudio/Modifiers/PreviewImages/A_Landing.gif)

- **Login Page**  
  ![Login](NoCodeStudio/Modifiers/PreviewImages/B_LoginPage.png)

- **Workspace Overview**  
  ![Workspace](NoCodeStudio/Modifiers/PreviewImages/C_Workspace.png)

- **Model Creation / Builder Page**  
  ![Builder](NoCodeStudio/Modifiers/PreviewImages/D_CreationPage.png)

- **Training Complete**  
  ![Training Finished](NoCodeStudio/Modifiers/PreviewImages/E_TrainingFinished.png)

- **Training Speed & Metrics**  
  ![Training Speed](NoCodeStudio/Modifiers/PreviewImages/F_TrainingSpeed.png)

---

## 📂 Project Structure

```
NoCodeStudio/
│
├── README.md                             # THIS FILE
│
├── apps/                                  # Webpage Core Files
│   │
│   ├── api/                               # API Files
│   │   ├── backend_api.ts                 # Main Express server
│   │   ├── package.json                   # API dependencies
│   │   ├── tsconfig.json                  # TypeScript config for the API service
│   │   └── .env                           # Environment variables (Replace with your MongoDB)
│   │
│   ├── py-tools/                          # Main Service Files
│   │   ├── main.py                        # FastAPI server
│   │   ├── parser.py                      # Torch layer discovery
│   │   ├── trainer.py                     # Full training engine
│   │   └── data/                          # Default Datasets (CIFAR-10 & MNIST)
│   │
│   └── web/                               # Frontend Files
│       ├── index.html                     # App root template
│       ├── package.json                   # Frontend dependencies
│       ├── vite.config.ts                 # Vite bundler setup
│       ├── tsconfig*.json                 # TypeScript configs
│       │
│       └── src/                           # Frontend UI
│           ├── App.tsx                    # Root React component
│           ├── main.tsx                   # ReactDOM entry
│           │
│           ├── assets/                    # UI Assets
│           ├── lib/                       # Axios instance
│           ├── store/                     # Authentication Files
│           └── pages/                     # Entire UI Experience Files
│
├── packages/                              # Shared cross-workspace TypeScript utilities
│
└── Modifiers/                             # Helper files for custom uploads and installation

```

---

## 🛠️ Tech Stack

### Frontend (Web Client)
Built with a modern React/Vite stack focused on speed, modularity, and smooth interaction:

**React 18 (TypeScript):** Core UI framework powering a fully component-based interface  
**Vite:** High-performance bundler enabling instant HMR and optimized production builds  
**Axios:** Typed HTTP client used for interacting with both backend services  
**Framer Motion:** Provides smooth transitions, animations, and micro-interactions  
**Lucide Icons / Tailwind (optional):** Clean, lightweight iconography and utility-first styling  

### Backend (Node API)
A TypeScript-based server responsible for authentication, resource management, and orchestration:

**Node.js + Express:** Main REST API layer handling auth, CRUD logic, and routing  
**MongoDB + Mongoose:** Document database storing users, projects, templates, and metadata  
**JWT Authentication:** Secure token-based authentication system  
**Express File Upload:** Enables uploading user-provided model and trainer files  

### ML Engine (Python Runtime)
A dedicated machine-learning service focused on training, inference, and PyTorch code integration:

**FastAPI:** High-performance Python framework serving the ML endpoints  
**PyTorch:** Deep learning framework used to generate, train, and run models  
**Torchvision:** Dataset utilities powering CIFAR-10 and MNIST loaders  
**Uvicorn:** ASGI server used to host the Python service  
**Custom Training Engine:** Auto-fixes shape mismatches, calculates metrics, manages threaded training, and provides ETA + speed reporting  


---

## 🔧 Setup & Run

Follow these steps to fully install and launch the NoCodeStudio platform (API + Web + Python ML Engine).

### 1) Configure the API Environment

Navigate to: `NoCodeStudio\apps\api\.env`

Inside this file, replace:
```
MONGO_URI=<your-mongodb-connection-string>
```

Use the MongoDB URL generated from your MongoDB Atlas project or local instance.

### 2) Create the Python Environment (Conda)

Go to: `NoCodeStudio\Modifiers\Create_nocodedl.bat`

Run this `.bat` file to automatically create a conda env and installs all Python dependencies.


### 3) Install Dependencies (PNPM)

Open Terminal and paste the following :
```bash
cd C:\NoCodeStudio\apps\api
pnpm install
cd C:\NoCodeStudio\apps\web
pnpm install
```
Then close terminal when done

### 4) Start All Services

You must run these in **three separate PowerShells**

- PowerShell 1 : Python ML Engine
```bash
conda activate nocodedl
cd C:\NoCodeStudio\apps\py-tools
uvicorn main:app --reload --port 8000
```
- PowerShell 2 : Node.js API
```bash
cd C:\NoCodeStudio\apps\api
pnpm dev
```
- Powershell 3 : Web Frontend
```bash
cd C:\NoCodeStudio\apps\web
pnpm dev
```

### 5) Open the Application

Open: `http://localhost:5173/`

You should now see the NoCodeStudio landing page with access to:
- Login  
- Workspace  
- Neural Network Builder  

*All three terminals must remain open while the platform is running.*

---

## 📡 API Overview

### Node API (Project & User Management)
Handles authentication, project storage, folder organization, and communication with the Python engine.

| Route              | Method | Description |
|--------------------|--------|-------------|
| `/auth/login`      | POST   | Authenticate user and return JWT token |
| `/auth/register`   | POST   | Create a new user account |
| `/api/projects`    | GET    | Fetch all projects for the authenticated user |
| `/api/projects`    | POST   | Create a new project (graph, metadata, thumbnail) |
| `/api/projects`    | DELETE | Delete a specific project |
| `/api/folders`     | CRUD   | Create, read, update, or delete user folders |
| `/api/templates`   | GET    | Fetch built-in project templates |
| `/api/upload`      | POST   | Upload custom model or trainer files |
| `/api/forward/*`   | ANY    | Proxy requests to Python ML Engine |

### Python Engine API (Training, Inference, Code Import/Export)
Executes ML tasks, converts graphs to PyTorch code, imports PyTorch models, and handles dataset-based training.

| Route            | Method | Description |
|------------------|--------|-------------|
| `/catalog`       | GET    | Returns all available PyTorch layers and metadata discovered via introspection |
| `/layer/{name}`  | GET    | Retrieves constructor parameters for a specific PyTorch layer |
| `/export`        | POST   | Converts a graph (nodes + edges) into a complete PyTorch `nn.Module` class |
| `/import`        | POST   | Parses user-provided PyTorch code and reconstructs an editable graph |
| `/api/train`     | POST   | Starts a training session (supports auto-fixing shapes, metrics, ETA updates) |
| `/api/run`       | POST   | Runs inference using a trained PyTorch model |
| `/cancel_training` | POST | Stops an active training process |
| `/health`        | GET    | Health check endpoint |

---

## 🩹 Troubleshooting

-   Check `axios.ts` base URLs if services do not connect
-   Ensure MongoDB is updated with your database and can append data
-   Ensure Python engine port matches frontend
-   Auto-fixing handles most layer mismatches but use custom trainer scripts to work on your datasets

------------------------------------------------------------------------

## 🧩 Customization

- You can upload your modified **training code** or even **model imports** via the UI interface in Builder mode.
- To Modify, use template in `NoCodeStudio\Modifiers`

- Datasets can be swapped or extended easily.

---

## ⚠️ Notes

- This is a local hosted run version only.
- Containerized version is separate and not on this repo!