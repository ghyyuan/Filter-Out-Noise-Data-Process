#!/usr/bin/env python3
"""
Invalid Review Detection System

This script merges review data with business metadata and detects various types
of anomalous/invalid reviews based on temporal, account, and spatial patterns.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2
from collections import defaultdict, Counter
import warnings

warnings.filterwarnings('ignore')


def load_data(reviews_path, metadata_path):
    """Load and parse JSON data files."""
    print("Loading reviews data...")
    reviews = []
    with open(reviews_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                reviews.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    
    print("Loading metadata...")
    metadata = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                metadata.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    
    # Convert to DataFrames
    reviews_df = pd.DataFrame(reviews)
    metadata_df = pd.DataFrame(metadata)
    
    print(f"Loaded {len(reviews_df)} reviews and {len(metadata_df)} businesses")
    return reviews_df, metadata_df


def merge_data(reviews_df, metadata_df):
    """Merge reviews with metadata on gmap_id."""
    print("Merging reviews with metadata...")
    
    # Remove duplicates from metadata first
    metadata_clean = metadata_df.drop_duplicates(subset=['gmap_id'])
    
    # Merge on gmap_id
    merged_df = reviews_df.merge(
        metadata_clean[['gmap_id', 'name', 'latitude', 'longitude', 'category', 
                       'avg_rating', 'num_of_reviews']],
        on='gmap_id',
        how='left',
        suffixes=('_review', '_business')
    )
    
    # Clean up column names
    merged_df = merged_df.rename(columns={
        'name_review': 'reviewer_name',
        'name_business': 'business_name'
    })
    
    # Convert time to datetime
    merged_df['datetime'] = pd.to_datetime(merged_df['time'], unit='ms')
    
    # Clean category column (handle lists and nulls)
    merged_df['category'] = merged_df['category'].apply(
        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'Unknown'
    )
    
    print(f"Merged dataset has {len(merged_df)} reviews")
    return merged_df


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate haversine distance between two points in kilometers."""
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return float('inf')
    
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c


def check_duplicate_same_biz(df):
    """Detect duplicate reviews: same user, same business, same timestamp."""
    print("Checking for duplicate same business reviews...")
    
    duplicates = df.duplicated(subset=['user_id', 'gmap_id', 'time'], keep=False)
    
    # Stronger evidence if business has very few reviews
    low_review_businesses = df['num_of_reviews'] < 10
    
    flags = duplicates.copy()
    reasons = pd.Series([''] * len(df), index=df.index)
    
    duplicate_indices = df[duplicates].index
    for idx in duplicate_indices:
        if low_review_businesses.iloc[idx]:
            reasons.iloc[idx] = 'duplicate same biz (low review count)'
        else:
            reasons.iloc[idx] = 'duplicate same biz'
    
    return flags, reasons


def check_multi_business_same_time(df):
    """Detect same user reviewing multiple businesses at same time."""
    print("Checking for multi-business same time reviews...")
    
    flags = pd.Series([False] * len(df), index=df.index)
    reasons = pd.Series([''] * len(df), index=df.index)
    
    # Group by user and time
    grouped = df.groupby(['user_id', 'time'])
    
    for (user_id, timestamp), group in grouped:
        if len(group) > 1:  # Multiple businesses at same time
            indices = group.index
            
            # Check if businesses are far apart
            coords = group[['latitude', 'longitude']].dropna()
            far_apart = False
            
            if len(coords) > 1:
                for i in range(len(coords)):
                    for j in range(i + 1, len(coords)):
                        dist = haversine_distance(
                            coords.iloc[i]['latitude'], coords.iloc[i]['longitude'],
                            coords.iloc[j]['latitude'], coords.iloc[j]['longitude']
                        )
                        if dist > 10:  # More than 10km apart
                            far_apart = True
                            break
                    if far_apart:
                        break
            
            # Check if same category
            categories = group['category'].unique()
            same_category = len(categories) == 1 and categories[0] != 'Unknown'
            
            reason = 'multi business same time'
            if far_apart:
                reason += ' (far apart)'
            elif same_category:
                reason += f' (same category: {categories[0]})'
            
            flags.loc[indices] = True
            reasons.loc[indices] = reason
    
    return flags, reasons


def check_burst_reviews(df, time_window_hours=24, min_reviews=10):
    """Detect burst of reviews in short time window."""
    print("Checking for burst reviews...")
    
    flags = pd.Series([False] * len(df), index=df.index)
    reasons = pd.Series([''] * len(df), index=df.index)
    
    # Sort by user and time
    df_sorted = df.sort_values(['user_id', 'datetime']).copy()
    
    for user_id, user_group in df_sorted.groupby('user_id'):
        user_group = user_group.copy()
        
        for i in range(len(user_group)):
            start_time = user_group.iloc[i]['datetime']
            end_time = start_time + timedelta(hours=time_window_hours)
            
            # Count reviews in time window
            window_reviews = user_group[
                (user_group['datetime'] >= start_time) &
                (user_group['datetime'] <= end_time)
            ]
            
            if len(window_reviews) >= min_reviews:
                # Check if mostly same category
                categories = window_reviews['category'].value_counts()
                dominant_category = categories.iloc[0] if len(categories) > 0 else 0
                same_category_pct = dominant_category / len(window_reviews)
                
                # Check geographic spread
                coords = window_reviews[['latitude', 'longitude']].dropna()
                different_locations = False
                if len(coords) > 1:
                    max_dist = 0
                    for ii in range(len(coords)):
                        for jj in range(ii + 1, len(coords)):
                            dist = haversine_distance(
                                coords.iloc[ii]['latitude'], coords.iloc[ii]['longitude'],
                                coords.iloc[jj]['latitude'], coords.iloc[jj]['longitude']
                            )
                            max_dist = max(max_dist, dist)
                    different_locations = max_dist > 100  # >100km apart
                
                reason = f'burst ({len(window_reviews)} reviews in {time_window_hours}h)'
                if same_category_pct > 0.7:
                    reason += f' (mostly {categories.index[0]})'
                if different_locations:
                    reason += ' (different locations)'
                
                indices = window_reviews.index
                flags.loc[indices] = True
                reasons.loc[indices] = reason
    
    return flags, reasons


def check_extreme_rating_patterns(df):
    """Detect users with extreme rating patterns."""
    print("Checking for extreme rating patterns...")
    
    flags = pd.Series([False] * len(df), index=df.index)
    reasons = pd.Series([''] * len(df), index=df.index)
    
    for user_id, user_group in df.groupby('user_id'):
        ratings = user_group['rating']
        
        # Only 1-star reviews (malicious)
        if len(ratings) >= 5 and all(ratings == 1):
            indices = user_group.index
            flags.loc[indices] = True
            reasons.loc[indices] = 'extreme rating (only 1-star)'
        
        # Only 5-star reviews (spam)
        elif len(ratings) >= 5 and all(ratings == 5):
            indices = user_group.index
            flags.loc[indices] = True
            reasons.loc[indices] = 'extreme rating (only 5-star)'
        
        # Check for ratings very different from business average
        for idx, row in user_group.iterrows():
            if pd.notna(row['avg_rating']):
                rating_diff = abs(row['rating'] - row['avg_rating'])
                if rating_diff >= 3:  # 3+ star difference
                    if not flags.loc[idx]:  # Don't overwrite existing flags
                        flags.loc[idx] = True
                        reasons.loc[idx] = f'extreme rating (diff from avg: {rating_diff:.1f})'
    
    return flags, reasons


def check_mass_reviews(df, min_reviews=50):
    """Detect users with too many reviews overall."""
    print("Checking for mass reviews...")
    
    flags = pd.Series([False] * len(df), index=df.index)
    reasons = pd.Series([''] * len(df), index=df.index)
    
    user_review_counts = df['user_id'].value_counts()
    mass_reviewers = user_review_counts[user_review_counts >= min_reviews].index
    
    for user_id in mass_reviewers:
        user_group = df[df['user_id'] == user_id]
        
        # Check if mostly reviewing low-review businesses
        low_review_businesses = (user_group['num_of_reviews'] < 10).sum()
        low_review_pct = low_review_businesses / len(user_group)
        
        reason = f'mass reviews ({len(user_group)} total)'
        if low_review_pct > 0.5:
            reason += f' (mostly low-review businesses: {low_review_pct:.1%})'
        
        indices = user_group.index
        flags.loc[indices] = True
        reasons.loc[indices] = reason
    
    return flags, reasons


def check_teleportation(df, max_speed_kmh=500):
    """Detect impossible travel speeds between reviews."""
    print("Checking for teleportation...")
    
    flags = pd.Series([False] * len(df), index=df.index)
    reasons = pd.Series([''] * len(df), index=df.index)
    
    # Sort by user and time
    df_sorted = df.sort_values(['user_id', 'datetime']).copy()
    
    for user_id, user_group in df_sorted.groupby('user_id'):
        user_group = user_group.copy()
        user_indices = list(user_group.index)
        
        for i in range(len(user_group) - 1):
            current_idx = user_indices[i]
            next_idx = user_indices[i + 1]
            
            current = user_group.loc[current_idx]
            next_review = user_group.loc[next_idx]
            
            # Calculate distance
            distance = haversine_distance(
                current['latitude'], current['longitude'],
                next_review['latitude'], next_review['longitude']
            )
            
            # Calculate time difference in hours
            time_diff = (next_review['datetime'] - current['datetime']).total_seconds() / 3600
            
            if time_diff > 0 and distance != float('inf'):
                speed = distance / time_diff
                
                if speed > max_speed_kmh:
                    reason = f'teleportation ({speed:.0f} km/h, {distance:.0f}km in {time_diff:.1f}h)'
                    
                    flags.loc[current_idx] = True
                    flags.loc[next_idx] = True
                    reasons.loc[current_idx] = reason
                    reasons.loc[next_idx] = reason
    
    return flags, reasons


def detect_invalid_reviews(reviews_path, metadata_path, output_path):
    """Main function to detect invalid reviews."""
    print("Starting invalid review detection...")
    
    # Load and merge data
    reviews_df, metadata_df = load_data(reviews_path, metadata_path)
    merged_df = merge_data(reviews_df, metadata_df)
    
    # Initialize flags and reasons
    is_invalid = pd.Series([False] * len(merged_df), index=merged_df.index)
    reasons = pd.Series([''] * len(merged_df), index=merged_df.index)
    
    # Run all anomaly detection functions
    anomaly_functions = [
        check_duplicate_same_biz,
        check_multi_business_same_time,
        check_burst_reviews,
        check_extreme_rating_patterns,
        check_mass_reviews,
        check_teleportation
    ]
    
    for func in anomaly_functions:
        try:
            flags, func_reasons = func(merged_df)
            
            # Combine flags (OR operation)
            is_invalid = is_invalid | flags
            
            # Combine reasons
            for idx in merged_df.index:
                if flags.iloc[idx] and func_reasons.iloc[idx]:
                    if reasons.iloc[idx]:
                        reasons.iloc[idx] += '; ' + func_reasons.iloc[idx]
                    else:
                        reasons.iloc[idx] = func_reasons.iloc[idx]
        
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
            continue
    
    # Add flags to dataframe
    merged_df['is_invalid'] = is_invalid
    merged_df['reason'] = reasons
    
    # Create output with business metadata
    output_df = merged_df[[
        'user_id', 'reviewer_name', 'time', 'datetime', 'rating', 'text',
        'gmap_id', 'business_name', 'latitude', 'longitude', 'category',
        'avg_rating', 'num_of_reviews', 'is_invalid', 'reason'
    ]].copy()
    
    # Save results
    print(f"Saving results to {output_path}...")
    output_df.to_csv(output_path, index=False)
    
    # Print summary
    total_reviews = len(output_df)
    invalid_reviews = output_df['is_invalid'].sum()
    
    print(f"\nSummary:")
    print(f"Total reviews: {total_reviews:,}")
    print(f"Invalid reviews: {invalid_reviews:,} ({invalid_reviews/total_reviews:.1%})")
    
    if invalid_reviews > 0:
        print(f"\nInvalid review breakdown:")
        reason_counts = Counter()
        for reason in output_df[output_df['is_invalid']]['reason']:
            for r in reason.split('; '):
                if r:
                    reason_counts[r.split(' (')[0]] += 1
        
        for reason, count in reason_counts.most_common():
            print(f"  {reason}: {count:,}")
    
    print(f"\nResults saved to: {output_path}")
    return output_df


if __name__ == "__main__":
    # Set file paths
    reviews_path = "data/reviews.json"
    metadata_path = "data/metadata.json"
    output_path = "data/metadata_with_flags.csv"
    
    # Run detection
    results = detect_invalid_reviews(reviews_path, metadata_path, output_path)