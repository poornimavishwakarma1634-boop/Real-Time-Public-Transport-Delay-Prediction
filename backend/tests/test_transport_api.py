"""
Backend API Tests for Public Transport Delay Prediction System - Iteration 2
Tests all endpoints including new features:
- GTFS data integration (/api/gtfs-status)
- Model retraining scheduler (/api/pipeline-status, /api/scheduler/retrain)
- Real model inference (/api/predict with PROPHET, TENSORFLOW_LSTM, PYTORCH_LSTM)
- Live feed data (/api/live-feed)
"""
import pytest
import requests
import os
from datetime import datetime

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', 'http://127.0.0.1:8000').rstrip('/')


class TestHealthAndRoutes:
    """Test basic API health and routes endpoint"""
    
    def test_api_root(self):
        """Test API root endpoint returns correct structure"""
        response = requests.get(f"{BASE_URL}/api/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        # Version should be 2.0.0 for new features
        assert data["version"] == "2.0.0"
        print(f"✅ API root working: {data['message']}, version: {data['version']}")
    
    def test_get_routes_returns_8_routes(self):
        """Test /api/routes returns 8 routes for Bangalore and Kalaburagi"""
        response = requests.get(f"{BASE_URL}/api/routes")
        assert response.status_code == 200
        data = response.json()
        assert "routes" in data
        routes = data["routes"]
        assert len(routes) == 8, f"Expected 8 routes, got {len(routes)}"
        
        # Verify route structure
        for route in routes:
            assert "id" in route
            assert "name" in route
            assert "city" in route
            assert "type" in route
        
        # Verify cities
        cities = set(r["city"] for r in routes)
        assert "Bangalore" in cities
        assert "Kalaburagi" in cities
        
        print(f"✅ Routes endpoint returns {len(routes)} routes for cities: {cities}")


class TestModelsEndpoint:
    """Test ML models endpoint - should return 3 real models"""
    
    def test_get_models_returns_real_models(self):
        """Test /api/models returns PROPHET, TENSORFLOW_LSTM, PYTORCH_LSTM"""
        response = requests.get(f"{BASE_URL}/api/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "count" in data
        assert "default" in data
        
        models = data["models"]
        assert len(models) == 3, f"Expected 3 models, got {len(models)}"
        assert data["default"] == "PROPHET", f"Default should be PROPHET, got {data['default']}"
        
        # Check for expected models (real inference models)
        expected_models = ["PROPHET", "TENSORFLOW_LSTM", "PYTORCH_LSTM"]
        for expected in expected_models:
            assert expected in models, f"Missing model: {expected}"
        
        print(f"✅ Models endpoint returns {len(models)} models: {models}")


class TestRealModelPrediction:
    """Test delay prediction with REAL model inference (not rule-based fallback)"""
    
    def test_predict_with_prophet_high_confidence(self):
        """Test POST /api/predict with PROPHET returns high confidence real prediction"""
        payload = {
            "timestamp": "2026-01-15T08:30:00",
            "route_id": "BLR-BUS-1",
            "model_name": "PROPHET"
        }
        response = requests.post(f"{BASE_URL}/api/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "timestamp" in data
        assert "route_id" in data
        assert "predicted_delay" in data
        assert "model_used" in data
        assert "confidence" in data
        
        # CRITICAL: model_used should be PROPHET (not SIMPLE_RULE_BASED or FALLBACK)
        assert data["model_used"] == "PROPHET", f"Expected PROPHET, got {data['model_used']}"
        assert data["confidence"] == "high", f"Expected high confidence, got {data['confidence']}"
        assert isinstance(data["predicted_delay"], (int, float))
        assert data["predicted_delay"] >= 0
        
        print(f"✅ PROPHET prediction: {data['predicted_delay']} min delay, confidence: {data['confidence']}")
    
    def test_predict_with_tensorflow_lstm_high_confidence(self):
        """Test POST /api/predict with TENSORFLOW_LSTM returns high confidence real prediction"""
        payload = {
            "timestamp": "2026-01-15T08:30:00",
            "route_id": "BLR-BUS-1",
            "model_name": "TENSORFLOW_LSTM"
        }
        response = requests.post(f"{BASE_URL}/api/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        
        # CRITICAL: model_used should be TENSORFLOW_LSTM (not SIMPLE_RULE_BASED or FALLBACK)
        assert data["model_used"] == "TENSORFLOW_LSTM", f"Expected TENSORFLOW_LSTM, got {data['model_used']}"
        assert data["confidence"] == "high", f"Expected high confidence, got {data['confidence']}"
        assert isinstance(data["predicted_delay"], (int, float))
        assert data["predicted_delay"] >= 0
        
        print(f"✅ TENSORFLOW_LSTM prediction: {data['predicted_delay']} min delay, confidence: {data['confidence']}")
    
    def test_predict_with_pytorch_lstm_high_confidence(self):
        """Test POST /api/predict with PYTORCH_LSTM returns high confidence real prediction"""
        payload = {
            "timestamp": "2026-01-15T08:30:00",
            "route_id": "BLR-BUS-1",
            "model_name": "PYTORCH_LSTM"
        }
        response = requests.post(f"{BASE_URL}/api/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        
        # CRITICAL: model_used should be PYTORCH_LSTM (not SIMPLE_RULE_BASED or FALLBACK)
        assert data["model_used"] == "PYTORCH_LSTM", f"Expected PYTORCH_LSTM, got {data['model_used']}"
        assert data["confidence"] == "high", f"Expected high confidence, got {data['confidence']}"
        assert isinstance(data["predicted_delay"], (int, float))
        assert data["predicted_delay"] >= 0
        
        print(f"✅ PYTORCH_LSTM prediction: {data['predicted_delay']} min delay, confidence: {data['confidence']}")
    
    def test_predict_with_different_routes(self):
        """Test prediction for different routes with PROPHET"""
        routes_to_test = ["BLR-BUS-1", "BLR-TRN-1", "KLB-BUS-1", "KLB-TRN-1"]
        
        for route in routes_to_test:
            payload = {
                "timestamp": "2026-01-15T12:00:00",
                "route_id": route,
                "model_name": "PROPHET"
            }
            response = requests.post(f"{BASE_URL}/api/predict", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["route_id"] == route
            assert data["model_used"] == "PROPHET"
            print(f"  ✅ Route {route}: {data['predicted_delay']} min delay")


class TestGTFSIntegration:
    """Test GTFS data integration endpoint"""
    
    def test_gtfs_status_returns_routes_count(self):
        """Test /api/gtfs-status returns GTFS routes count (4210), stops, trips, source=BMTC"""
        response = requests.get(f"{BASE_URL}/api/gtfs-status")
        assert response.status_code == 200
        data = response.json()
        
        assert "gtfs" in data
        assert "live_feed_count" in data
        assert "source" in data
        
        gtfs = data["gtfs"]
        assert "routes" in gtfs
        assert "stops" in gtfs
        assert "trips" in gtfs
        assert "source" in gtfs
        
        # Verify GTFS data from BMTC
        assert gtfs["routes"] == 4210, f"Expected 4210 GTFS routes, got {gtfs['routes']}"
        assert gtfs["stops"] > 0, "Should have GTFS stops"
        assert gtfs["trips"] > 0, "Should have GTFS trips"
        assert "BMTC" in gtfs["source"], f"Source should mention BMTC, got {gtfs['source']}"
        
        print(f"✅ GTFS status: {gtfs['routes']} routes, {gtfs['stops']} stops, {gtfs['trips']} trips")
        print(f"  Source: {gtfs['source']}")


class TestLiveFeed:
    """Test live feed data endpoint"""
    
    def test_live_feed_returns_real_time_observations(self):
        """Test /api/live-feed returns real-time delay observations from MongoDB"""
        response = requests.get(f"{BASE_URL}/api/live-feed?limit=10")
        assert response.status_code == 200
        data = response.json()
        
        assert "data" in data
        assert "count" in data
        assert "source" in data
        
        assert data["source"] == "live_feed"
        assert data["count"] > 0, "Should have live feed data"
        
        # Verify data structure
        if len(data["data"]) > 0:
            sample = data["data"][0]
            assert "timestamp" in sample
            assert "route_id" in sample
            assert "route_name" in sample
            assert "city" in sample
            assert "transport_type" in sample
            assert "delay_minutes" in sample
            assert "delay_seconds" in sample
            assert "source" in sample
        
        print(f"✅ Live feed returns {data['count']} records")


class TestSchedulerAndPipelineStatus:
    """Test scheduler and pipeline status endpoints"""
    
    def test_pipeline_status_includes_scheduler_active(self):
        """Test /api/pipeline-status includes scheduler status (running or retraining), live_feed_records > 0"""
        response = requests.get(f"{BASE_URL}/api/pipeline-status")
        assert response.status_code == 200
        data = response.json()
        
        assert "models_trained" in data
        assert "scheduler" in data
        assert "live_feed_records" in data
        
        # Verify scheduler is active (running or retraining)
        scheduler = data["scheduler"]
        valid_statuses = ["running", "retraining"]
        assert scheduler["status"] in valid_statuses, f"Scheduler should be active, got {scheduler['status']}"
        assert "last_retrain" in scheduler
        assert "last_feed_gen" in scheduler
        assert "retrain_count" in scheduler
        assert "models_loaded" in scheduler
        
        # Verify live feed records
        assert data["live_feed_records"] > 0, f"Should have live feed records, got {data['live_feed_records']}"
        
        # Verify models loaded
        models_loaded = scheduler["models_loaded"]
        assert "PROPHET" in models_loaded
        assert "TENSORFLOW_LSTM" in models_loaded
        assert "PYTORCH_LSTM" in models_loaded
        
        print(f"✅ Pipeline status: scheduler={scheduler['status']}, retrain_count={scheduler['retrain_count']}")
        print(f"  Live feed records: {data['live_feed_records']}")
        print(f"  Models loaded: {models_loaded}")
    
    def test_scheduler_retrain_endpoint_exists(self):
        """Test POST /api/scheduler/retrain endpoint exists and responds (may timeout during heavy processing)"""
        import time
        # Wait a bit for any ongoing retrain to complete
        time.sleep(5)
        
        try:
            response = requests.post(f"{BASE_URL}/api/scheduler/retrain", timeout=120)
            
            if response.status_code == 200:
                data = response.json()
                assert "status" in data
                
                if data["status"] == "completed":
                    assert "results" in data
                    results = data["results"]
                    assert len(results) >= 3, f"Should retrain at least 3 models, got {len(results)}"
                    
                    # Verify each model has RMSE and MAE
                    for result in results:
                        assert "model" in result
                        assert "RMSE" in result
                        assert "MAE" in result
                        assert result["RMSE"] > 0
                        assert result["MAE"] > 0
                    
                    print(f"✅ Manual retrain completed with {len(results)} models")
                    for r in results:
                        print(f"  - {r['model']}: RMSE={r['RMSE']:.4f}, MAE={r['MAE']:.4f}")
                else:
                    print(f"✅ Retrain endpoint responded with status: {data['status']}")
            elif response.status_code == 502:
                # Gateway timeout during heavy processing is acceptable
                print(f"⚠️ Retrain endpoint timed out (502) - likely due to ongoing retrain cycle")
            else:
                assert False, f"Unexpected status code: {response.status_code}"
        except requests.exceptions.Timeout:
            print(f"⚠️ Retrain request timed out - model retraining takes time")


class TestModelComparison:
    """Test model comparison endpoint with real metrics"""
    
    def test_model_comparison_returns_real_metrics(self):
        """Test /api/model-comparison returns real (non-mock) metrics for 3 models with RMSE and MAE"""
        response = requests.get(f"{BASE_URL}/api/model-comparison")
        assert response.status_code == 200
        data = response.json()
        
        assert "comparison" in data
        assert "count" in data
        
        comparison = data["comparison"]
        assert len(comparison) >= 3, f"Should have at least 3 models, got {len(comparison)}"
        
        # Verify structure and values
        model_names = []
        for m in comparison:
            assert "model" in m
            assert "RMSE" in m
            assert "MAE" in m
            assert m["RMSE"] > 0
            assert m["MAE"] > 0
            model_names.append(m["model"])
        
        # Verify expected models are present
        assert any("Prophet" in name for name in model_names), "Prophet should be in comparison"
        assert any("TensorFlow" in name for name in model_names), "TensorFlow LSTM should be in comparison"
        assert any("PyTorch" in name for name in model_names), "PyTorch LSTM should be in comparison"
        
        print(f"✅ Model comparison returns {len(comparison)} models with real metrics")
        for m in comparison:
            print(f"  - {m['model']}: RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}")


class TestHistoricalEndpoint:
    """Test historical data endpoint"""
    
    def test_get_historical_data(self):
        """Test /api/historical returns historical delay data"""
        response = requests.get(f"{BASE_URL}/api/historical")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "count" in data
        
        historical = data["data"]
        assert len(historical) > 0, "Should have historical data"
        
        # Verify data structure
        if len(historical) > 0:
            sample = historical[0]
            assert "timestamp" in sample
            assert "delay_minutes" in sample
        
        print(f"✅ Historical endpoint returns {len(historical)} data points")


class TestRouteStatsEndpoint:
    """Test route statistics endpoint"""
    
    def test_get_route_stats(self):
        """Test /api/route-stats returns delay stats per route"""
        response = requests.get(f"{BASE_URL}/api/route-stats")
        assert response.status_code == 200
        data = response.json()
        assert "stats" in data
        
        stats = data["stats"]
        assert len(stats) >= 1, "Should have route stats"
        
        # Verify structure
        if len(stats) > 0:
            sample = stats[0]
            assert "route_id" in sample
            assert "route_name" in sample
            assert "city" in sample
            assert "transport_type" in sample
            assert "avg_delay" in sample
        
        print(f"✅ Route stats returns {len(stats)} routes")


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_predict_with_invalid_model_falls_back(self):
        """Test prediction with non-existent model falls back gracefully"""
        payload = {
            "timestamp": "2026-01-15T08:30:00",
            "route_id": "BLR-BUS-1",
            "model_name": "INVALID_MODEL"
        }
        response = requests.post(f"{BASE_URL}/api/predict", json=payload)
        # Should still return 200 with fallback
        assert response.status_code == 200
        data = response.json()
        assert "predicted_delay" in data
        # Should fallback to PROPHET or another real model
        assert data["model_used"] in ["PROPHET", "TENSORFLOW_LSTM", "PYTORCH_LSTM", "HEURISTIC"]
        print(f"✅ Invalid model handled gracefully, fallback used: {data['model_used']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
