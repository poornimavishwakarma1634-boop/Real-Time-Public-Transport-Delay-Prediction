"""
GTFS Data Fetcher & Live Feed Simulator
Downloads BMTC/NEKRTC static GTFS data, processes it,
and generates live delay observations stored in MongoDB.
"""
import os
import io
import csv
import zipfile
import logging
import asyncio
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger(__name__)

# BMTC unofficial GTFS source (Vonter/bmtc-gtfs on GitHub)
BMTC_GTFS_URL = "https://github.com/Vonter/bmtc-gtfs/raw/main/gtfs/bmtc.zip"
GTFS_CACHE_DIR = Path("/app/ml_project/data/gtfs")

# Route mapping from GTFS route_ids to our internal IDs
ROUTE_PROFILES = {
    "BLR-BUS-1": {"city": "Bangalore", "type": "Bus", "base_delay": 3.0, "rush_mult": 3.5,
                   "name": "Majestic to Koramangala (Bus)", "gtfs_pattern": "KBS-"},
    "BLR-BUS-2": {"city": "Bangalore", "type": "Bus", "base_delay": 4.0, "rush_mult": 4.0,
                   "name": "Whitefield to Electronic City (Bus)", "gtfs_pattern": "ITPL-"},
    "BLR-TRN-1": {"city": "Bangalore", "type": "Train", "base_delay": 1.5, "rush_mult": 1.5,
                   "name": "Namma Metro Purple Line (Train)", "gtfs_pattern": "PURPLE"},
    "BLR-TRN-2": {"city": "Bangalore", "type": "Train", "base_delay": 1.2, "rush_mult": 1.2,
                   "name": "Namma Metro Green Line (Train)", "gtfs_pattern": "GREEN"},
    "KLB-BUS-1": {"city": "Kalaburagi", "type": "Bus", "base_delay": 5.0, "rush_mult": 2.5,
                   "name": "Kalaburagi City Bus Ring Road (Bus)", "gtfs_pattern": "KLB-RING"},
    "KLB-BUS-2": {"city": "Kalaburagi", "type": "Bus", "base_delay": 6.0, "rush_mult": 3.0,
                   "name": "NEKRTC to Gulbarga University (Bus)", "gtfs_pattern": "KLB-UNI"},
    "KLB-TRN-1": {"city": "Kalaburagi", "type": "Train", "base_delay": 8.0, "rush_mult": 2.0,
                   "name": "Kalaburagi - Bangalore Express (Train)", "gtfs_pattern": "KLB-BLR"},
    "BLR-KLB-EXP": {"city": "Intercity", "type": "Train", "base_delay": 10.0, "rush_mult": 2.5,
                     "name": "Bangalore to Kalaburagi Intercity Express (Train)", "gtfs_pattern": "BLR-KLB"},
}


class GTFSFetcher:
    """Downloads and processes GTFS data from BMTC/NEKRTC feeds"""

    def __init__(self):
        GTFS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.gtfs_routes = []
        self.gtfs_stops = []
        self.gtfs_trips = []
        self.last_fetch = None

    def fetch_bmtc_gtfs(self):
        """Download and extract BMTC GTFS zip"""
        logger.info("Fetching BMTC GTFS feed from GitHub...")
        try:
            resp = requests.get(BMTC_GTFS_URL, timeout=30)
            if resp.status_code == 200:
                zf = zipfile.ZipFile(io.BytesIO(resp.content))
                zf.extractall(GTFS_CACHE_DIR / "bmtc")
                self.last_fetch = datetime.now(timezone.utc)
                logger.info(f"BMTC GTFS extracted to {GTFS_CACHE_DIR / 'bmtc'}")
                self._parse_gtfs(GTFS_CACHE_DIR / "bmtc")
                return True
            else:
                logger.warning(f"GTFS fetch returned {resp.status_code}")
                return False
        except Exception as e:
            logger.warning(f"GTFS fetch failed: {e}. Using cached/generated data.")
            return False

    def _parse_gtfs(self, gtfs_dir):
        """Parse GTFS text files into memory"""
        try:
            routes_file = gtfs_dir / "routes.txt"
            if routes_file.exists():
                self.gtfs_routes = pd.read_csv(routes_file).to_dict("records")
                logger.info(f"Parsed {len(self.gtfs_routes)} GTFS routes")

            stops_file = gtfs_dir / "stops.txt"
            if stops_file.exists():
                self.gtfs_stops = pd.read_csv(stops_file).head(200).to_dict("records")
                logger.info(f"Parsed {len(self.gtfs_stops)} GTFS stops")

            trips_file = gtfs_dir / "trips.txt"
            if trips_file.exists():
                self.gtfs_trips = pd.read_csv(trips_file).head(500).to_dict("records")
                logger.info(f"Parsed {len(self.gtfs_trips)} GTFS trips")
        except Exception as e:
            logger.warning(f"GTFS parse error: {e}")

    def get_gtfs_summary(self):
        return {
            "routes": len(self.gtfs_routes),
            "stops": len(self.gtfs_stops),
            "trips": len(self.gtfs_trips),
            "last_fetch": self.last_fetch.isoformat() if self.last_fetch else None,
            "source": "BMTC GitHub (Vonter/bmtc-gtfs)"
        }


class LiveFeedSimulator:
    """
    Generates realistic real-time delay observations based on
    route profiles and temporal patterns. Stores data in MongoDB.
    """

    def __init__(self, db):
        self.db = db
        self.collection = db["live_delays"]
        self._running = False

    def _generate_delay(self, route_id: str, ts: datetime) -> float:
        """Generate a realistic delay in minutes for a given route and timestamp"""
        profile = ROUTE_PROFILES.get(route_id, ROUTE_PROFILES["BLR-BUS-1"])
        base = profile["base_delay"]
        hour = ts.hour
        dow = ts.weekday()

        # Rush hour effect
        if profile["type"] == "Bus":
            if 8 <= hour < 10:
                time_eff = np.random.normal(profile["rush_mult"] * 2, profile["rush_mult"] * 0.6)
            elif 17 <= hour < 20:
                time_eff = np.random.normal(profile["rush_mult"] * 2.5, profile["rush_mult"] * 0.8)
            elif 23 <= hour or hour < 5:
                time_eff = np.random.normal(0.3, 0.2)
            else:
                time_eff = np.random.normal(profile["rush_mult"] * 0.8, 1.0)
        else:  # Train
            if 8 <= hour < 10 or 17 <= hour < 19:
                time_eff = np.random.normal(profile["rush_mult"], 0.8)
            else:
                time_eff = np.random.normal(0.5, 0.4)

        # Weekend reduction
        day_eff = np.random.normal(-1.5, 0.5) if dow >= 5 else np.random.normal(0.5, 0.3)

        # Monsoon (Jun-Sep) weather effect
        month = ts.month
        if month in (6, 7, 8, 9) and np.random.random() < 0.45:
            weather_eff = np.random.normal(3, 1.5)
        elif month in (10, 11) and np.random.random() < 0.25:
            weather_eff = np.random.normal(2, 1)
        else:
            weather_eff = 0

        # Random incident 3%
        incident = np.random.uniform(10, 25) if np.random.random() < 0.03 else 0

        total = base + time_eff + day_eff + weather_eff + incident
        return round(max(0, total), 2)

    async def generate_batch(self, minutes_back: int = 15):
        """Generate a batch of live delay observations and insert into MongoDB"""
        now = datetime.now(timezone.utc)
        records = []
        for route_id, profile in ROUTE_PROFILES.items():
            delay = self._generate_delay(route_id, now)
            rec = {
                "timestamp": now.isoformat(),
                "route_id": route_id,
                "route_name": profile["name"],
                "city": profile["city"],
                "transport_type": profile["type"],
                "delay_minutes": delay,
                "delay_seconds": round(delay * 60, 1),
                "source": "live_feed",
                "created_at": now.isoformat()
            }
            records.append(rec)

        if records:
            await self.collection.insert_many(records)
            logger.info(f"Inserted {len(records)} live delay records")
        return records

    def generate_batch_sync(self):
        """Synchronous version for background thread (uses pymongo directly)"""
        from pymongo import MongoClient
        sync_client = MongoClient(os.environ.get('MONGO_URL', 'mongodb://localhost:27017'))
        sync_db = sync_client[os.environ.get('DB_NAME', 'test_database')]
        coll = sync_db["live_delays"]

        now = datetime.now(timezone.utc)
        records = []
        for route_id, profile in ROUTE_PROFILES.items():
            delay = self._generate_delay(route_id, now)
            records.append({
                "timestamp": now.isoformat(),
                "route_id": route_id,
                "route_name": profile["name"],
                "city": profile["city"],
                "transport_type": profile["type"],
                "delay_minutes": delay,
                "delay_seconds": round(delay * 60, 1),
                "source": "live_feed",
                "created_at": now.isoformat()
            })
        if records:
            coll.insert_many(records)
            logger.info(f"Inserted {len(records)} live records (sync)")
        sync_client.close()
        return records

    async def get_recent(self, limit=500):
        """Get recent live delay observations"""
        cursor = self.collection.find(
            {}, {"_id": 0}
        ).sort("timestamp", -1).limit(limit)
        return await cursor.to_list(limit)

    async def get_count(self):
        return await self.collection.count_documents({})

    async def backfill(self, hours=24):
        """Backfill historical live data for the last N hours"""
        count = await self.get_count()
        if count > 100:
            logger.info(f"Skipping backfill, already have {count} records")
            return count

        logger.info(f"Backfilling {hours} hours of live data...")
        now = datetime.now(timezone.utc)
        records = []
        for mins_ago in range(0, hours * 60, 15):  # every 15 min
            ts = now - timedelta(minutes=mins_ago)
            for route_id, profile in ROUTE_PROFILES.items():
                delay = self._generate_delay(route_id, ts)
                records.append({
                    "timestamp": ts.isoformat(),
                    "route_id": route_id,
                    "route_name": profile["name"],
                    "city": profile["city"],
                    "transport_type": profile["type"],
                    "delay_minutes": delay,
                    "delay_seconds": round(delay * 60, 1),
                    "source": "backfill",
                    "created_at": now.isoformat()
                })

        if records:
            await self.collection.insert_many(records)
            logger.info(f"Backfilled {len(records)} records ({hours}h)")
        return len(records)
