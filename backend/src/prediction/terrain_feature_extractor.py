"""
DEM-Based Terrain Feature Extractor for Rockfall Prediction
=========================================================

This module extracts critical terrain features from Digital Elevation Models (DEM)
that are used as input features for rockfall risk prediction.

Features Extracted:
- Slope (primary geometric risk factor)
- Aspect (slope direction)
- Curvature (surface curvature analysis)
- Roughness (surface irregularity)
- Slope Variability (local terrain complexity)
- Topographic Wetness Index (water accumulation effects)
- Terrain Ruggedness Index (overall terrain complexity)
- Instability Index (combined assessment score)
"""

import numpy as np
import pandas as pd
import rasterio
import rasterio.features
from rasterio.windows import Window
from scipy import ndimage
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class DEMTerrainExtractor:
    """Extract terrain features from DEM files for rockfall prediction"""
    
    def __init__(self):
        self.dem_data = None
        self.transform = None
        self.crs = None
        self.nodata = None
        self.terrain_features = {}
        
    def load_dem(self, dem_path: str) -> Dict:
        """Load DEM file and extract basic information"""
        
        try:
            with rasterio.open(dem_path) as src:
                self.dem_data = src.read(1).astype(np.float64)
                self.transform = src.transform
                self.crs = src.crs
                self.nodata = src.nodata
                
                # Replace nodata values with NaN
                if self.nodata is not None:
                    self.dem_data[self.dem_data == self.nodata] = np.nan
                
                dem_info = {
                    'shape': self.dem_data.shape,
                    'bounds': src.bounds,
                    'crs': str(src.crs),
                    'resolution': (self.transform[0], abs(self.transform[4])),
                    'elevation_range': (np.nanmin(self.dem_data), np.nanmax(self.dem_data)),
                    'valid_pixels': np.sum(~np.isnan(self.dem_data))
                }
                
                print(f"✅ DEM loaded successfully: {dem_path}")
                print(f"📏 Shape: {dem_info['shape']}")
                print(f"🏔️ Elevation range: {dem_info['elevation_range'][0]:.1f}m to {dem_info['elevation_range'][1]:.1f}m")
                print(f"📊 Valid pixels: {dem_info['valid_pixels']:,}")
                
                return dem_info
                
        except Exception as e:
            print(f"❌ Error loading DEM: {e}")
            raise
    
    def calculate_slope_aspect(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate slope and aspect from DEM using gradient analysis"""
        
        if self.dem_data is None:
            raise ValueError("DEM data not loaded. Call load_dem() first.")
        
        # Calculate gradients (partial derivatives)
        pixel_size_x = abs(self.transform[0])
        pixel_size_y = abs(self.transform[4])
        
        # Use Sobel operators for gradient calculation
        grad_x = ndimage.sobel(self.dem_data, axis=1) / (8 * pixel_size_x)
        grad_y = ndimage.sobel(self.dem_data, axis=0) / (8 * pixel_size_y)
        
        # Calculate slope in degrees
        slope_radians = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
        slope_degrees = np.degrees(slope_radians)
        
        # Calculate aspect in degrees (0-360)
        aspect_radians = np.arctan2(-grad_y, grad_x)
        aspect_degrees = np.degrees(aspect_radians)
        aspect_degrees = (aspect_degrees + 360) % 360  # Ensure 0-360 range
        
        # Handle flat areas (slope = 0) - set aspect to -1
        aspect_degrees[slope_degrees < 0.1] = -1
        
        self.terrain_features['slope'] = slope_degrees
        self.terrain_features['aspect'] = aspect_degrees
        
        print(f"✅ Slope calculated - Range: {np.nanmin(slope_degrees):.1f}° to {np.nanmax(slope_degrees):.1f}°")
        print(f"✅ Aspect calculated - {np.sum(aspect_degrees >= 0):,} valid direction pixels")
        
        return slope_degrees, aspect_degrees
    
    def calculate_curvature(self) -> Dict[str, np.ndarray]:
        """Calculate profile, planform, and total curvature"""
        
        if self.dem_data is None:
            raise ValueError("DEM data not loaded. Call load_dem() first.")
        
        pixel_size = abs(self.transform[0])  # Assuming square pixels
        
        # Second derivatives using finite differences
        d2z_dx2 = ndimage.generic_filter(self.dem_data, 
                                        lambda x: x[0] - 2*x[1] + x[2] if len(x) == 3 else 0,
                                        size=(1, 3)) / (pixel_size**2)
        
        d2z_dy2 = ndimage.generic_filter(self.dem_data,
                                        lambda x: x[0] - 2*x[1] + x[2] if len(x) == 3 else 0,
                                        size=(3, 1)) / (pixel_size**2)
        
        d2z_dxdy = ndimage.generic_filter(self.dem_data,
                                         lambda x: (x[0] + x[2] - x[1] - x[3])/4 if len(x) >= 4 else 0,
                                         footprint=np.array([[1,0,1],[0,0,0],[1,0,1]])) / (pixel_size**2)
        
        # First derivatives (already calculated for slope)
        grad_x = ndimage.sobel(self.dem_data, axis=1) / (8 * pixel_size)
        grad_y = ndimage.sobel(self.dem_data, axis=0) / (8 * pixel_size)
        
        # Calculate curvatures
        p = grad_x**2
        q = grad_y**2
        r = d2z_dx2
        s = d2z_dxdy
        t = d2z_dy2
        
        # Profile curvature (curvature in the direction of steepest slope)
        profile_curvature = -(r*p + 2*s*grad_x*grad_y + t*q) / ((p + q) * (1 + p + q)**1.5)
        
        # Planform curvature (curvature perpendicular to steepest slope)
        planform_curvature = -(t*p - 2*s*grad_x*grad_y + r*q) / ((p + q)**1.5)
        
        # Total curvature (mean curvature)
        total_curvature = -(r + t) / (2 * (1 + p + q)**1.5)
        
        # Handle division by zero and invalid values
        profile_curvature[np.isnan(profile_curvature) | np.isinf(profile_curvature)] = 0
        planform_curvature[np.isnan(planform_curvature) | np.isinf(planform_curvature)] = 0
        total_curvature[np.isnan(total_curvature) | np.isinf(total_curvature)] = 0
        
        curvatures = {
            'profile_curvature': profile_curvature,
            'planform_curvature': planform_curvature,
            'total_curvature': total_curvature
        }
        
        self.terrain_features.update(curvatures)
        
        print(f"✅ Curvature calculated:")
        for name, curv in curvatures.items():
            print(f"   {name}: {np.nanmin(curv):.4f} to {np.nanmax(curv):.4f}")
        
        return curvatures
    
    def calculate_roughness(self, window_size: int = 3) -> np.ndarray:
        """Calculate terrain roughness using standard deviation of elevation"""
        
        if self.dem_data is None:
            raise ValueError("DEM data not loaded. Call load_dem() first.")
        
        # Calculate local standard deviation of elevation
        def local_std(window):
            return np.std(window) if not np.all(np.isnan(window)) else 0
        
        roughness = ndimage.generic_filter(
            self.dem_data, 
            local_std, 
            size=window_size
        )
        
        # Normalize roughness to 0-1 scale
        roughness_normalized = (roughness - np.nanmin(roughness)) / (np.nanmax(roughness) - np.nanmin(roughness))
        roughness_normalized[np.isnan(roughness_normalized)] = 0
        
        self.terrain_features['roughness'] = roughness_normalized
        
        print(f"✅ Roughness calculated - Range: {np.nanmin(roughness):.2f} to {np.nanmax(roughness):.2f}")
        
        return roughness_normalized
    
    def calculate_slope_variability(self, window_size: int = 5) -> np.ndarray:
        """Calculate local variability of slope (standard deviation of slope)"""
        
        if 'slope' not in self.terrain_features:
            self.calculate_slope_aspect()
        
        slope = self.terrain_features['slope']
        
        # Calculate local standard deviation of slope
        def local_slope_std(window):
            return np.std(window) if not np.all(np.isnan(window)) else 0
        
        slope_variability = ndimage.generic_filter(
            slope, 
            local_slope_std, 
            size=window_size
        )
        
        self.terrain_features['slope_variability'] = slope_variability
        
        print(f"✅ Slope variability calculated - Range: {np.nanmin(slope_variability):.2f}° to {np.nanmax(slope_variability):.2f}°")
        
        return slope_variability
    
    def calculate_topographic_wetness_index(self) -> np.ndarray:
        """Calculate Topographic Wetness Index (TWI)"""
        
        if 'slope' not in self.terrain_features:
            self.calculate_slope_aspect()
        
        slope = self.terrain_features['slope']
        
        # Estimate contributing area using flow accumulation
        # Simplified approach: use elevation-based flow direction
        
        # Calculate flow direction (D8 algorithm simplified)
        flow_dir = np.zeros_like(self.dem_data)
        rows, cols = self.dem_data.shape
        
        # For each cell, find direction of steepest descent
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                if np.isnan(self.dem_data[i, j]):
                    continue
                
                center_elev = self.dem_data[i, j]
                max_slope = 0
                steepest_dir = 0
                
                # Check 8 neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols and not np.isnan(self.dem_data[ni, nj]):
                            neighbor_elev = self.dem_data[ni, nj]
                            if neighbor_elev < center_elev:
                                slope_to_neighbor = (center_elev - neighbor_elev) / np.sqrt(di**2 + dj**2)
                                if slope_to_neighbor > max_slope:
                                    max_slope = slope_to_neighbor
                                    steepest_dir = di * 3 + dj + 4  # Direction encoding
                
                flow_dir[i, j] = steepest_dir
        
        # Simplified contributing area calculation
        # In practice, this would require complex flow accumulation algorithms
        # For now, use a proxy based on local topography
        contributing_area = np.ones_like(self.dem_data) * abs(self.transform[0]) * abs(self.transform[4])
        
        # Estimate contributing area based on local elevation and curvature
        if 'total_curvature' in self.terrain_features:
            curvature = self.terrain_features['total_curvature']
            # Areas with negative curvature (concave) tend to accumulate more water
            area_multiplier = np.where(curvature < 0, 1 + abs(curvature) * 10, 1)
            contributing_area *= area_multiplier
        
        # Calculate TWI
        slope_radians = np.radians(slope + 0.01)  # Add small value to avoid division by zero
        twi = np.log(contributing_area / np.tan(slope_radians))
        
        # Handle invalid values
        twi[np.isnan(twi) | np.isinf(twi)] = 0
        
        self.terrain_features['wetness_index'] = twi
        
        print(f"✅ Topographic Wetness Index calculated - Range: {np.nanmin(twi):.2f} to {np.nanmax(twi):.2f}")
        
        return twi
    
    def calculate_terrain_ruggedness_index(self, window_size: int = 3) -> np.ndarray:
        """Calculate Terrain Ruggedness Index (TRI)"""
        
        if self.dem_data is None:
            raise ValueError("DEM data not loaded. Call load_dem() first.")
        
        # Calculate mean elevation difference in neighborhood
        def local_elevation_diff(window):
            if np.all(np.isnan(window)):
                return 0
            center = window[len(window)//2]
            if np.isnan(center):
                return 0
            return np.nanmean(np.abs(window - center))
        
        tri = ndimage.generic_filter(
            self.dem_data, 
            local_elevation_diff, 
            size=window_size
        )
        
        self.terrain_features['terrain_ruggedness_index'] = tri
        
        print(f"✅ Terrain Ruggedness Index calculated - Range: {np.nanmin(tri):.2f} to {np.nanmax(tri):.2f}")
        
        return tri
    
    def calculate_instability_index(self) -> np.ndarray:
        """Calculate combined terrain instability index"""
        
        # Ensure all required features are calculated
        required_features = ['slope', 'roughness']
        
        for feature in required_features:
            if feature not in self.terrain_features:
                if feature == 'slope':
                    self.calculate_slope_aspect()
                elif feature == 'roughness':
                    self.calculate_roughness()
        
        # Calculate additional features if not present
        if 'total_curvature' not in self.terrain_features:
            self.calculate_curvature()
        if 'slope_variability' not in self.terrain_features:
            self.calculate_slope_variability()
        if 'terrain_ruggedness_index' not in self.terrain_features:
            self.calculate_terrain_ruggedness_index()
        
        # Normalize features to 0-1 scale
        slope_norm = self.terrain_features['slope'] / 90  # Normalize slope to 0-1
        roughness_norm = self.terrain_features['roughness']  # Already normalized
        
        # Normalize curvature (handle negative values)
        curvature = self.terrain_features['total_curvature']
        curvature_abs = np.abs(curvature)
        curvature_norm = curvature_abs / (np.nanmax(curvature_abs) + 1e-10)
        
        # Normalize slope variability
        slope_var = self.terrain_features['slope_variability']
        slope_var_norm = slope_var / (np.nanmax(slope_var) + 1e-10)
        
        # Normalize terrain ruggedness
        tri = self.terrain_features['terrain_ruggedness_index']
        tri_norm = tri / (np.nanmax(tri) + 1e-10)
        
        # Calculate weighted instability index
        instability = (
            slope_norm * 0.40 +          # Slope is most important
            roughness_norm * 0.20 +      # Surface roughness
            curvature_norm * 0.15 +      # Surface curvature
            slope_var_norm * 0.15 +      # Local slope variability
            tri_norm * 0.10              # Terrain ruggedness
        )
        
        # Ensure valid range
        instability = np.clip(instability, 0, 1)
        instability[np.isnan(instability)] = 0
        
        self.terrain_features['instability_index'] = instability
        
        print(f"✅ Instability Index calculated - Range: {np.nanmin(instability):.3f} to {np.nanmax(instability):.3f}")
        
        return instability
    
    def extract_all_features(self) -> Dict[str, np.ndarray]:
        """Extract all terrain features from DEM"""
        
        print("🏔️ Extracting all terrain features from DEM...")
        
        # Calculate all features
        self.calculate_slope_aspect()
        self.calculate_curvature()
        self.calculate_roughness()
        self.calculate_slope_variability()
        self.calculate_topographic_wetness_index()
        self.calculate_terrain_ruggedness_index()
        self.calculate_instability_index()
        
        print(f"✅ All terrain features extracted! Total: {len(self.terrain_features)} features")
        
        return self.terrain_features
    
    def sample_features_for_prediction(self, n_samples: int = 1000, 
                                     method: str = 'random') -> pd.DataFrame:
        """Sample feature points for prediction model training"""
        
        if not self.terrain_features:
            self.extract_all_features()
        
        rows, cols = self.dem_data.shape
        
        if method == 'random':
            # Random sampling
            valid_mask = ~np.isnan(self.dem_data)
            valid_indices = np.where(valid_mask)
            
            if len(valid_indices[0]) < n_samples:
                n_samples = len(valid_indices[0])
                print(f"⚠️ Reducing sample size to {n_samples} (all valid pixels)")
            
            sample_indices = np.random.choice(len(valid_indices[0]), n_samples, replace=False)
            sample_rows = valid_indices[0][sample_indices]
            sample_cols = valid_indices[1][sample_indices]
            
        elif method == 'stratified':
            # Stratified sampling based on slope
            slope = self.terrain_features['slope']
            valid_mask = ~np.isnan(slope)
            
            # Create slope bins
            slope_bins = [0, 15, 30, 45, 60, 90]
            sample_rows = []
            sample_cols = []
            
            samples_per_bin = n_samples // len(slope_bins)
            
            for i in range(len(slope_bins) - 1):
                bin_mask = valid_mask & (slope >= slope_bins[i]) & (slope < slope_bins[i+1])
                bin_indices = np.where(bin_mask)
                
                if len(bin_indices[0]) > 0:
                    n_bin_samples = min(samples_per_bin, len(bin_indices[0]))
                    selected = np.random.choice(len(bin_indices[0]), n_bin_samples, replace=False)
                    sample_rows.extend(bin_indices[0][selected])
                    sample_cols.extend(bin_indices[1][selected])
            
            sample_rows = np.array(sample_rows)
            sample_cols = np.array(sample_cols)
            
        else:
            raise ValueError("Method must be 'random' or 'stratified'")
        
        # Extract feature values at sample points
        sample_data = {}
        
        # Add spatial coordinates
        sample_data['x'] = sample_cols * abs(self.transform[0]) + self.transform[2]
        sample_data['y'] = sample_rows * self.transform[4] + self.transform[5]
        sample_data['elevation'] = self.dem_data[sample_rows, sample_cols]
        
        # Extract all terrain features
        for feature_name, feature_array in self.terrain_features.items():
            sample_data[feature_name] = feature_array[sample_rows, sample_cols]
        
        # Create DataFrame
        df = pd.DataFrame(sample_data)
        
        # Remove any remaining NaN values
        df = df.dropna()
        
        print(f"✅ Sampled {len(df)} feature points using {method} method")
        print(f"📊 Features available: {list(df.columns)}")
        
        return df
    
    def visualize_terrain_features(self, save_path: str = None):
        """Create comprehensive visualization of extracted terrain features"""
        
        if not self.terrain_features:
            print("❌ No terrain features found. Run extract_all_features() first.")
            return
        
        # Select key features for visualization
        viz_features = ['slope', 'aspect', 'total_curvature', 'roughness', 
                       'slope_variability', 'wetness_index', 'terrain_ruggedness_index', 
                       'instability_index']
        
        available_features = [f for f in viz_features if f in self.terrain_features]
        n_features = len(available_features)
        
        # Create subplots
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('DEM-Derived Terrain Features for Rockfall Prediction', 
                    fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for i, feature_name in enumerate(available_features):
            if i >= len(axes):
                break
                
            feature_data = self.terrain_features[feature_name]
            
            # Create appropriate visualization based on feature
            if feature_name == 'aspect':
                # Use circular colormap for aspect
                im = axes[i].imshow(feature_data, cmap='hsv', vmin=0, vmax=360)
            elif feature_name in ['total_curvature']:
                # Use diverging colormap for curvature
                vmax = np.nanmax(np.abs(feature_data))
                im = axes[i].imshow(feature_data, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            else:
                # Use sequential colormap for other features
                im = axes[i].imshow(feature_data, cmap='viridis')
            
            axes[i].set_title(f'{feature_name.replace("_", " ").title()}')
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            
            # Add colorbar
            plt.colorbar(im, ax=axes[i], shrink=0.8)
        
        # Hide unused subplots
        for i in range(len(available_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Terrain features visualization saved to: {save_path}")
        
        plt.show()
    
    def get_feature_statistics(self) -> Dict:
        """Get comprehensive statistics of extracted terrain features"""
        
        if not self.terrain_features:
            print("❌ No terrain features found. Run extract_all_features() first.")
            return {}
        
        statistics = {}
        
        for feature_name, feature_array in self.terrain_features.items():
            valid_data = feature_array[~np.isnan(feature_array)]
            
            if len(valid_data) > 0:
                statistics[feature_name] = {
                    'count': len(valid_data),
                    'mean': float(np.mean(valid_data)),
                    'std': float(np.std(valid_data)),
                    'min': float(np.min(valid_data)),
                    'max': float(np.max(valid_data)),
                    'median': float(np.median(valid_data)),
                    'percentile_25': float(np.percentile(valid_data, 25)),
                    'percentile_75': float(np.percentile(valid_data, 75))
                }
        
        return statistics


def main():
    """Demonstrate terrain feature extraction from DEM"""
    
    # Initialize extractor
    extractor = DEMTerrainExtractor()
    
    # Test with available DEM file
    dem_path = "data/DEM/Bingham_Canyon_Mine.tif"
    
    try:
        # Load DEM
        dem_info = extractor.load_dem(dem_path)
        
        # Extract all features
        features = extractor.extract_all_features()
        
        # Get feature statistics
        stats = extractor.get_feature_statistics()
        
        # Sample features for prediction
        sample_df = extractor.sample_features_for_prediction(n_samples=2000, method='stratified')
        
        # Save sampled data
        output_path = "outputs/dem_terrain_features.csv"
        sample_df.to_csv(output_path, index=False)
        print(f"💾 Terrain features saved to: {output_path}")
        
        # Create visualizations
        viz_path = "outputs/terrain_features_visualization.png"
        extractor.visualize_terrain_features(viz_path)
        
        # Print statistics
        print("\n📊 TERRAIN FEATURE STATISTICS")
        print("="*50)
        for feature, stat in stats.items():
            print(f"\n{feature.upper().replace('_', ' ')}:")
            print(f"  Count: {stat['count']:,}")
            print(f"  Mean: {stat['mean']:.3f}")
            print(f"  Std: {stat['std']:.3f}")
            print(f"  Range: {stat['min']:.3f} to {stat['max']:.3f}")
        
        return sample_df, stats
        
    except Exception as e:
        print(f"❌ Error processing DEM: {e}")
        return None, None


if __name__ == "__main__":
    terrain_data, statistics = main()