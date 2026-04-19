"""
Step 1: Data Ingestion - Bangalore & Kalaburagi Public Transport
Creates realistic transport delay data for bus and train routes
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import print_section_header


ROUTES = {
    'BLR-BUS-1': {'name': 'Bangalore Majestic to Koramangala (Bus)', 'city': 'Bangalore', 'type': 'Bus', 'base_delay': 3.0},
    'BLR-BUS-2': {'name': 'Bangalore Whitefield to Electronic City (Bus)', 'city': 'Bangalore', 'type': 'Bus', 'base_delay': 4.0},
    'BLR-TRN-1': {'name': 'Bangalore Namma Metro Purple Line (Train)', 'city': 'Bangalore', 'type': 'Train', 'base_delay': 1.5},
    'BLR-TRN-2': {'name': 'Bangalore Namma Metro Green Line (Train)', 'city': 'Bangalore', 'type': 'Train', 'base_delay': 1.2},
    'KLB-BUS-1': {'name': 'Kalaburagi City Bus - Ring Road (Bus)', 'city': 'Kalaburagi', 'type': 'Bus', 'base_delay': 5.0},
    'KLB-BUS-2': {'name': 'Kalaburagi NEKRTC to Gulbarga University (Bus)', 'city': 'Kalaburagi', 'type': 'Bus', 'base_delay': 6.0},
    'KLB-TRN-1': {'name': 'Kalaburagi - Bangalore Express (Train)', 'city': 'Kalaburagi', 'type': 'Train', 'base_delay': 8.0},
    'BLR-KLB-EXP': {'name': 'Bangalore to Kalaburagi Intercity Express (Train)', 'city': 'Intercity', 'type': 'Train', 'base_delay': 10.0},
}

STOPS = {
    'Bangalore': ['Majestic', 'Koramangala', 'Whitefield', 'Electronic City', 'Indiranagar', 
                  'Jayanagar', 'MG Road', 'Yeshwantpur', 'Hebbal', 'Banashankari'],
    'Kalaburagi': ['Kalaburagi Central', 'Gulbarga University', 'Supermarket', 'Jagat Circle',
                   'Aland Road', 'Sedam Road', 'NEKRTC Bus Stand', 'Railway Station'],
    'Intercity': ['Bangalore City', 'Yadgir', 'Raichur', 'Kalaburagi']
}


def generate_transport_data(num_days=90, freq_minutes=15):
    """
    Generate synthetic transport delay data for Bangalore & Kalaburagi
    freq_minutes=15 for 15-minute intervals (96 per day)
    """
    print_section_header("Generating Bangalore & Kalaburagi Transport Data")
    
    np.random.seed(42)
    samples_per_day = int(24 * 60 / freq_minutes)
    
    start_date = datetime(2025, 10, 1)
    timestamps = [start_date + timedelta(minutes=freq_minutes * i) 
                  for i in range(num_days * samples_per_day)]
    
    all_records = []
    
    for ts in timestamps:
        route_id = np.random.choice(list(ROUTES.keys()))
        route_info = ROUTES[route_id]
        city = route_info['city'] if route_info['city'] != 'Intercity' else np.random.choice(['Bangalore', 'Kalaburagi'])
        stops = STOPS.get(route_info['city'], STOPS['Bangalore'])
        
        hour = ts.hour
        day_of_week = ts.weekday()
        
        base_delay = route_info['base_delay']
        
        # Bangalore traffic is notorious during rush hours
        if route_info['city'] == 'Bangalore' and route_info['type'] == 'Bus':
            if 8 <= hour < 10:   # Morning rush
                time_effect = np.random.normal(8, 3)
            elif 17 <= hour < 20: # Evening rush - Bangalore traffic peaks
                time_effect = np.random.normal(12, 4)
            elif 23 <= hour or hour < 5:
                time_effect = np.random.normal(0.5, 0.3)
            else:
                time_effect = np.random.normal(3, 1.5)
        elif route_info['city'] == 'Kalaburagi' and route_info['type'] == 'Bus':
            if 8 <= hour < 10:
                time_effect = np.random.normal(5, 2)
            elif 17 <= hour < 19:
                time_effect = np.random.normal(6, 2.5)
            elif 23 <= hour or hour < 5:
                time_effect = np.random.normal(0.3, 0.2)
            else:
                time_effect = np.random.normal(2, 1)
        elif route_info['type'] == 'Train':
            # Trains are generally more reliable
            if 8 <= hour < 10 or 17 <= hour < 19:
                time_effect = np.random.normal(2, 1)
            else:
                time_effect = np.random.normal(0.5, 0.5)
            # Intercity trains have their own pattern
            if route_info['city'] == 'Intercity':
                time_effect += np.random.normal(5, 3)
        else:
            time_effect = np.random.normal(2, 1)
        
        # Weekend effect
        if day_of_week >= 5:
            day_effect = np.random.normal(-2, 1)  # Less traffic on weekends
        else:
            day_effect = np.random.normal(1, 0.5)
        
        # Rain effect (monsoon months Jul-Sep more rain)
        month = ts.month
        if month in [6, 7, 8, 9]:  # Monsoon season in Karnataka
            rain_prob = 0.5
        elif month in [10, 11]:     # Post-monsoon
            rain_prob = 0.3
        else:
            rain_prob = 0.15
        
        is_rain = 1 if np.random.random() < rain_prob else 0
        weather_effect = np.random.normal(3, 1.5) if is_rain else 0
        
        # Random incidents (accidents, breakdowns)
        if np.random.random() < 0.03:
            incident_effect = np.random.uniform(10, 25)
        else:
            incident_effect = 0
        
        total_delay = base_delay + time_effect + day_effect + weather_effect + incident_effect
        total_delay = max(0, total_delay)
        
        record = {
            'timestamp': ts,
            'route_id': route_id,
            'route_name': route_info['name'],
            'city': route_info['city'],
            'transport_type': route_info['type'],
            'stop_id': np.random.choice(stops),
            'vehicle_id': f"V-{route_id}-{np.random.randint(1, 6):02d}",
            'delay_seconds': total_delay * 60,
            'delay_minutes': round(total_delay, 2),
            'is_rain': is_rain,
            'is_weekend': 1 if day_of_week >= 5 else 0,
            'hour': hour,
            'day_of_week': day_of_week
        }
        all_records.append(record)
    
    df = pd.DataFrame(all_records)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"Generated {len(df)} records over {num_days} days")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nRoutes breakdown:")
    print(df['route_name'].value_counts().to_string())
    print(f"\nDelay Statistics (minutes):")
    print(df['delay_minutes'].describe())
    print(f"\nBy City:")
    print(df.groupby('city')['delay_minutes'].mean().to_string())
    print(f"\nBy Transport Type:")
    print(df.groupby('transport_type')['delay_minutes'].mean().to_string())
    
    return df


def save_raw_data(df, output_path):
    """Save raw data to CSV"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nRaw data saved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.2f} KB")


def main():
    print_section_header("BANGALORE & KALABURAGI TRANSPORT DATA INGESTION")
    
    df = generate_transport_data(num_days=90, freq_minutes=15)
    
    output_path = '/app/ml_project/data/raw/gtfs_data.csv'
    save_raw_data(df, output_path)
    
    print("\n" + "="*70)
    print("Data Ingestion Complete!")
    print("="*70)
    
    return df


if __name__ == "__main__":
    df = main()
