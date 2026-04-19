from fastapi import FastAPI, APIRouter, Query
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone

# Import services
from ml_service import ml_service
from gtfs_fetcher import GTFSFetcher, LiveFeedSimulator
from scheduler import RetrainingScheduler

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

app = FastAPI(title="TransitML — Public Transport Delay Prediction API")
api_router = APIRouter(prefix="/api")

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Globals (initialised at startup) ─────────────────────────────────────
gtfs_fetcher = GTFSFetcher()
live_feed = LiveFeedSimulator(db)
scheduler: Optional[RetrainingScheduler] = None


# ── Pydantic Models ──────────────────────────────────────────────────────
class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheckCreate(BaseModel):
    client_name: str

class PredictionRequest(BaseModel):
    timestamp: str = Field(..., description="ISO format timestamp")
    route_id: str = Field(default="BLR-BUS-1")
    model_name: str = Field(default="PROPHET")

class PredictionResponse(BaseModel):
    timestamp: str
    route_id: str
    predicted_delay: float
    model_used: str
    confidence: str


# ── Root / Health ────────────────────────────────────────────────────────
@api_router.get("/")
async def root():
    return {
        "message": "TransitML — Public Transport Delay Prediction API",
        "version": "2.0.0",
        "cities": ["Bangalore", "Kalaburagi"],
        "models": ml_service.get_available_models()
    }


# ── Prediction ───────────────────────────────────────────────────────────
@api_router.post("/predict", response_model=PredictionResponse)
async def predict_delay(request: PredictionRequest):
    prediction = ml_service.predict(
        timestamp=request.timestamp,
        route_id=request.route_id,
        model_name=request.model_name
    )
    return PredictionResponse(
        timestamp=request.timestamp,
        route_id=request.route_id,
        predicted_delay=prediction['predicted_delay'],
        model_used=prediction['model_used'],
        confidence=prediction['confidence']
    )


@api_router.get("/models")
async def get_available_models():
    models = ml_service.get_available_models()
    return {"models": models, "count": len(models), "default": "PROPHET"}


# ── Historical & Live Data ───────────────────────────────────────────────
@api_router.get("/historical")
async def get_historical_data(limit: int = Query(default=100, le=1000)):
    data = ml_service.get_historical_data(limit=limit)
    return {"data": data, "count": len(data)}


@api_router.get("/live-feed")
async def get_live_feed(limit: int = Query(default=100, le=500)):
    """Get recent live delay observations from MongoDB"""
    data = await live_feed.get_recent(limit)
    return {"data": data, "count": len(data), "source": "live_feed"}


# ── Model Comparison ─────────────────────────────────────────────────────
@api_router.get("/model-comparison")
async def get_model_comparison():
    comparison = ml_service.get_model_comparison()
    return {"comparison": comparison, "count": len(comparison)}


# ── Routes ───────────────────────────────────────────────────────────────
@api_router.get("/routes")
async def get_available_routes():
    return {
        "routes": [
            {"id": "BLR-BUS-1", "name": "Majestic to Koramangala (Bus)", "city": "Bangalore", "type": "Bus"},
            {"id": "BLR-BUS-2", "name": "Whitefield to Electronic City (Bus)", "city": "Bangalore", "type": "Bus"},
            {"id": "BLR-TRN-1", "name": "Namma Metro Purple Line (Train)", "city": "Bangalore", "type": "Train"},
            {"id": "BLR-TRN-2", "name": "Namma Metro Green Line (Train)", "city": "Bangalore", "type": "Train"},
            {"id": "KLB-BUS-1", "name": "Kalaburagi City Bus Ring Road (Bus)", "city": "Kalaburagi", "type": "Bus"},
            {"id": "KLB-BUS-2", "name": "NEKRTC to Gulbarga University (Bus)", "city": "Kalaburagi", "type": "Bus"},
            {"id": "KLB-TRN-1", "name": "Kalaburagi - Bangalore Express (Train)", "city": "Kalaburagi", "type": "Train"},
            {"id": "BLR-KLB-EXP", "name": "Bangalore to Kalaburagi Intercity (Train)", "city": "Intercity", "type": "Train"}
        ]
    }


# ── Route Stats ──────────────────────────────────────────────────────────
@api_router.get("/route-stats")
async def get_route_stats():
    try:
        import pandas as pd
        df = pd.read_csv('/app/ml_project/data/raw/gtfs_data.csv')
        stats = df.groupby(['route_id', 'route_name', 'city', 'transport_type'])['delay_minutes'].agg(
            ['mean', 'std', 'min', 'max', 'count']
        ).reset_index()
        stats.columns = ['route_id', 'route_name', 'city', 'transport_type',
                         'avg_delay', 'std_delay', 'min_delay', 'max_delay', 'count']
        return {"stats": stats.round(2).to_dict('records')}
    except Exception as e:
        return {"stats": [], "error": str(e)}


# ── Pipeline / Scheduler Status ──────────────────────────────────────────
@api_router.get("/pipeline-status")
async def get_pipeline_status():
    import glob
    models_found = []
    for ext in ('**/*.pkl', '**/*.keras', '**/*.pth'):
        for p in glob.glob(f'/app/ml_project/models/{ext}', recursive=True):
            models_found.append(os.path.basename(p))

    reports = [os.path.basename(p) for p in glob.glob('/app/ml_project/outputs/reports/*_results.txt')]
    viz = [os.path.basename(p) for p in glob.glob('/app/ml_project/outputs/visualizations/*.png')]

    sched_status = scheduler.get_status() if scheduler else {"status": "not_started"}
    live_count = await live_feed.get_count()

    return {
        "models_trained": models_found,
        "reports_generated": reports,
        "visualizations": viz,
        "pipeline_complete": len(models_found) >= 3,
        "scheduler": sched_status,
        "live_feed_records": live_count
    }


@api_router.get("/gtfs-status")
async def get_gtfs_status():
    """Get GTFS feed integration status"""
    return {
        "gtfs": gtfs_fetcher.get_gtfs_summary(),
        "live_feed_count": await live_feed.get_count(),
        "source": "BMTC GitHub (Vonter/bmtc-gtfs) + NEKRTC simulation"
    }


@api_router.post("/scheduler/retrain")
async def trigger_retrain():
    """Manually trigger a model retrain cycle"""
    from scheduler import retrain_all
    try:
        results = retrain_all()
        if results:
            ml_service.reload_models()
        return {"status": "completed", "results": results}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── Legacy endpoints ─────────────────────────────────────────────────────
@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.model_dump()
    status_obj = StatusCheck(**status_dict)
    doc = status_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    _ = await db.status_checks.insert_one(doc)
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find({}, {"_id": 0}).to_list(1000)
    for check in status_checks:
        if isinstance(check['timestamp'], str):
            check['timestamp'] = datetime.fromisoformat(check['timestamp'])
    return status_checks


# ── Include router ───────────────────────────────────────────────────────
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup / Shutdown ───────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global scheduler
    logger.info("=== TransitML Startup ===")

    # 1. Fetch GTFS data
    gtfs_fetcher.fetch_bmtc_gtfs()

    # 2. Backfill live feed if empty
    await live_feed.backfill(hours=48)

    # 3. Start scheduler (retrain every 60 min, feed every 15 min)
    scheduler = RetrainingScheduler(db, live_feed, ml_service, interval_minutes=60)
    scheduler.start()

    logger.info("=== Startup complete ===")


@app.on_event("shutdown")
async def shutdown():
    if scheduler:
        scheduler.stop()
    client.close()
