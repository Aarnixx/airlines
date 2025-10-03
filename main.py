import os
import sys
import time
import json
import math
import requests
import pandas as pd
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

API_KEY = "a67e8979758f4feea51637156bbdaf25"
csv_path = r"C:\Users\aarni\OneDrive\Desktop\Python\AirlineAnalytics\airlines_flights_data.csv"
cache_dir = r"C:\Users\aarni\OneDrive\Desktop\Python\AirlineAnalytics\cache"
output_csv_default = r"C:\Users\aarni\OneDrive\Desktop\Python\AirlineAnalytics\airlines_with_distances.csv"

os.makedirs(cache_dir, exist_ok=True)
city_cache_file = os.path.join(cache_dir, "city_coords.json")
pair_cache_file = os.path.join(cache_dir, "pair_distances.json")

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

def haversine_km(lat1, lon1, lat2, lon2):
    """Return distance in km between two lat/lon points using Haversine formula."""
    R = 6371.0088  # Earth radius (km)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * (2 * math.asin(math.sqrt(a)))

def load_json_or_empty(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def geoapify_get_city_coords(city, api_key=API_KEY, retry=3, pause=1.0):
    """
    Return (lat, lon) for a city string using Geoapify. Returns None on failure.
    Respects basic retry/backoff.
    """
    if not city or str(city).strip() == "":
        return None
    url = (
        "https://api.geoapify.com/v1/geocode/search"
        f"?text={requests.utils.quote(str(city))}&lang=en&limit=1&type=city&format=json&apiKey={api_key}"
    )
    for attempt in range(1, retry + 1):
        try:
            r = requests.get(url, timeout=12)
            if r.status_code == 200:
                j = r.json()
                if "results" in j and len(j["results"]) > 0:
                    res = j["results"][0]
                    lat = res.get("lat")
                    lon = res.get("lon")
                    if lat is not None and lon is not None:
                        return float(lat), float(lon)
                return None
            elif r.status_code in (429, 502, 503, 504):
                time.sleep(pause * attempt)
            else:
                time.sleep(pause)
        except requests.RequestException:
            time.sleep(pause * attempt)
    return None

class AirlineAnalytics:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.city_cache = load_json_or_empty(city_cache_file)  # city -> [lat, lon] tai None
        self.pair_cache = load_json_or_empty(pair_cache_file)  # "src/dst" -> distance_km tai None
        self.loaded = False

    def load_data(self):
        print("Loading CSV (this may take several seconds for large files)...")
        self.df = pd.read_csv(self.csv_path)
        required_cols = {"source_city", "destination_city", "airline"}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f"CSV must contain at least these columns: {required_cols}")
        self.df["source_city"] = self.df["source_city"].astype(str).str.strip()
        self.df["destination_city"] = self.df["destination_city"].astype(str).str.strip()
        self.loaded = True
        print(f"Loaded {len(self.df):,} rows.")

    def ensure_city_coords(self, verbose=False):
        if not self.loaded:
            self.load_data()
        unique_cities = set(self.df["source_city"].unique()) | set(self.df["destination_city"].unique())
        unique_cities = {c for c in unique_cities if c and str(c).strip() != ""}
        missing = [c for c in unique_cities if c not in self.city_cache or not self.city_cache[c]]
        print(f"{len(unique_cities):,} unique cities found, {len(missing):,} missing from cache.")
        if not missing:
            if verbose:
                print("All city coordinates available in cache.")
            return

        for i, city in enumerate(missing, 1):
            if verbose and (i % 50 == 0 or i == 1):
                print(f"Fetching coords: {i}/{len(missing)} - {city}")
            coords = geoapify_get_city_coords(city)
            if coords:
                self.city_cache[city] = [coords[0], coords[1]]
            else:
                self.city_cache[city] = None
            time.sleep(0.08)

        save_json(city_cache_file, self.city_cache)
        print(f"Saved city cache to: {city_cache_file}")

    def compute_pair_distances(self, verbose=False):
        if not self.loaded:
            self.load_data()
        pairs = self.df[["source_city", "destination_city"]].drop_duplicates()
        print(f"{len(pairs):,} unique city-pairs to consider.")
        new_count = 0
        for idx, row in pairs.iterrows():
            src = row["source_city"]
            dst = row["destination_city"]
            if not src or not dst:
                continue
            key = f"{src}||{dst}"
            if key in self.pair_cache and isinstance(self.pair_cache[key], (float, int)):
                continue
            src_coords = self.city_cache.get(src)
            dst_coords = self.city_cache.get(dst)
            if not src_coords or not dst_coords:
                self.pair_cache[key] = None
                continue
            dist_km = haversine_km(src_coords[0], src_coords[1], dst_coords[0], dst_coords[1])
            self.pair_cache[key] = round(dist_km, 2)
            new_count += 1
            if verbose and new_count % 200 == 0:
                print(f"Computed {new_count} new pair distances...")
        save_json(pair_cache_file, self.pair_cache)
        print(f"Pair distances cache saved to: {pair_cache_file} ({new_count} new)")

    def enrich_dataframe(self):
        if not self.loaded:
            self.load_data()

        def _map_lat(city):
            v = self.city_cache.get(city)
            return v[0] if v else None

        def _map_lon(city):
            v = self.city_cache.get(city)
            return v[1] if v else None

        self.df["source_lat"] = self.df["source_city"].map(_map_lat)
        self.df["source_lon"] = self.df["source_city"].map(_map_lon)
        self.df["dest_lat"] = self.df["destination_city"].map(_map_lat)
        self.df["dest_lon"] = self.df["destination_city"].map(_map_lon)

        def _map_pair_dist(row):
            key = f"{row['source_city']}||{row['destination_city']}"
            return self.pair_cache.get(key)

        self.df["distance_km"] = self.df.apply(_map_pair_dist, axis=1)

    def compute_summaries(self):
        if "distance_km" not in self.df.columns:
            self.enrich_dataframe()
        out = {}
        out["total_flights"] = len(self.df)
        out["total_airlines"] = int(self.df["airline"].nunique())
        out["flights_per_airline"] = self.df["airline"].value_counts().to_dict()
        out["most_visited_cities"] = self.df["destination_city"].value_counts().head(10).to_dict()

        distances = self.df["distance_km"].dropna()
        if not distances.empty:
            out["distance_mean_km"] = float(distances.mean())
            out["distance_median_km"] = float(distances.median())
            out["distance_max_km"] = float(distances.max())
            out["distance_min_km"] = float(distances.min())
        else:
            out["distance_mean_km"] = out["distance_median_km"] = out["distance_max_km"] = out["distance_min_km"] = None

        if "price" in self.df.columns:
            prices = pd.to_numeric(self.df["price"], errors="coerce").dropna()
            out["price_mean"] = float(prices.mean()) if not prices.empty else None
            out["price_median"] = float(prices.median()) if not prices.empty else None
            try:
                out["price_per_airline"] = self.df.groupby("airline")["price"].apply(lambda s: pd.to_numeric(s, errors="coerce").mean()).to_dict()
            except Exception:
                out["price_per_airline"] = {}
        else:
            out["price_mean"] = out["price_median"] = None
            out["price_per_airline"] = {}

        if "stops" in self.df.columns:
            out["stops_distribution"] = self.df["stops"].value_counts().to_dict()
        if "class" in self.df.columns:
            out["class_distribution"] = self.df["class"].value_counts().to_dict()
        if "days_left" in self.df.columns:
            try:
                out["days_left_mean"] = float(pd.to_numeric(self.df["days_left"], errors="coerce").mean())
            except Exception:
                out["days_left_mean"] = None

        per_airline = {}
        for airline, group in self.df.groupby("airline"):
            g = {}
            g["num_flights"] = int(len(group))
            dist_series = group["distance_km"].dropna()
            g["avg_distance_km"] = float(dist_series.mean()) if not dist_series.empty else None
            if "price" in group.columns:
                pseries = pd.to_numeric(group["price"], errors="coerce").dropna()
                g["avg_price"] = float(pseries.mean()) if not pseries.empty else None
            g["most_common_destinations"] = group["destination_city"].value_counts().head(5).to_dict()
            per_airline[airline] = g
        out["per_airline"] = per_airline
        return out

    def export_enriched_csv(self, path=output_csv_default):
        if "distance_km" not in self.df.columns:
            self.enrich_dataframe()
        self.df.to_csv(path, index=False)
        return path

class AirlineGUI:
    def __init__(self, root: ctk.CTk, analytics: AirlineAnalytics):
        self.root = root
        self.analytics = analytics
        self.df = analytics.df

        root.title("Airline Analytics - CTk")
        root.geometry("1200x820")
        root.minsize(900, 600)
        root.protocol("WM_DELETE_WINDOW", self.on_closing)

        top_frame = ctk.CTkFrame(root)
        top_frame.pack(side="top", fill="x", padx=10, pady=8)

        ctk.CTkLabel(top_frame, text="Airline:", width=60).pack(side="left", padx=(6, 4))
        self.airline_var = ctk.StringVar(value="")
        airlines = sorted(self.df["airline"].unique())
        self.airline_combo = ctk.CTkComboBox(top_frame, values=[""] + airlines, variable=self.airline_var, width=250)
        self.airline_combo.pack(side="left", padx=6)

        ctk.CTkButton(top_frame, text="Show Summary", command=self.show_summary).pack(side="left", padx=6)
        ctk.CTkButton(top_frame, text="Plot Metrics", command=self.plot_metrics).pack(side="left", padx=6)
        ctk.CTkButton(top_frame, text="Recompute caches", command=self.recompute_caches).pack(side="left", padx=6)
        ctk.CTkButton(top_frame, text="Export Enriched CSV", command=self.export_csv).pack(side="left", padx=6)
        ctk.CTkButton(top_frame, text="Quit", command=self.on_closing).pack(side="right", padx=6)

        middle_frame = ctk.CTkFrame(root)
        middle_frame.pack(side="top", fill="x", padx=10, pady=(0, 8))

        self.summary_box = ctk.CTkTextbox(middle_frame, height=220)
        self.summary_box.pack(side="left", fill="x", expand=True, padx=(0,6))

        info_frame = ctk.CTkFrame(middle_frame, width=240)
        info_frame.pack(side="right", fill="y")
        info_frame.pack_propagate(False)
        ctk.CTkLabel(info_frame, text="Notes", anchor="w").pack(anchor="nw", padx=8, pady=(6,2))
        info_text = (
            "- Uses cached coords/pair distances\n"
            "- Recompute caches only when needed\n"
            "- Export enriched CSV for ML\n"
            "- Large datasets supported (unique pair caching)"
        )
        ctk.CTkLabel(info_frame, text=info_text, wraplength=220, anchor="nw", justify="left").pack(padx=8, pady=4)

        self.plot_container = ctk.CTkScrollableFrame(root, label_text="Plots")
        self.plot_container.pack(side="top", fill="both", expand=True, padx=10, pady=(0,10))

        self.show_summary(global_view=True)

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you really want to quit?"):
            try:
                save_json(city_cache_file, self.analytics.city_cache)
                save_json(pair_cache_file, self.analytics.pair_cache)
            except Exception:
                pass
            self.root.destroy()
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)

    def show_summary(self, global_view=False):
        stats = self.analytics.compute_summaries()
        airline = self.airline_var.get().strip()
        self.summary_box.delete("0.0", "end")

        lines = []
        lines.append(f"Total flights: {stats['total_flights']:,}")
        lines.append(f"Distinct airlines: {stats['total_airlines']:,}")
        lines.append(f"Global average distance (km): {stats['distance_mean_km']}")
        lines.append(f"Global median distance (km): {stats['distance_median_km']}")
        if stats.get("price_mean") is not None:
            lines.append(f"Global average price: {stats['price_mean']:.2f}")
        lines.append("")
        lines.append("Top 10 most visited destinations:")
        for city, cnt in stats["most_visited_cities"].items():
            lines.append(f"  {city}: {cnt:,}")
        lines.append("")
        lines.append("Flights per airline (top 10):")
        for a, c in list(stats["flights_per_airline"].items())[:10]:
            lines.append(f"  {a}: {c:,}")
        if airline:
            pa = stats["per_airline"].get(airline)
            if pa:
                lines.append("")
                lines.append(f"--- Stats for {airline} ---")
                lines.append(f"Number of flights: {pa['num_flights']:,}")
                lines.append(f"Avg distance (km): {pa['avg_distance_km']}")
                lines.append(f"Avg price: {pa.get('avg_price')}")
                lines.append("Top destinations:")
                for d, k in pa["most_common_destinations"].items():
                    lines.append(f"  {d}: {k:,}")

        self.summary_box.insert("0.0", "\n".join(lines))

    def clear_plots(self):
        for widget in self.plot_container.winfo_children():
            widget.destroy()

    def plot_metrics(self):
        airline = self.airline_var.get().strip()
        df = self.df[self.df["airline"] == airline] if airline else self.df
        title_add = f" - {airline}" if airline else " - ALL"

        self.clear_plots()
        try:
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            d = df["distance_km"].dropna()
            if d.empty:
                ax1.text(0.5, 0.5, "No distance data available", ha="center")
            else:
                ax1.hist(d, bins=40)
            ax1.set_title(f"Distance distribution{title_add}")
            ax1.set_xlabel("Distance (km)")
            ax1.set_ylabel("Count")
            fig1.tight_layout()
            frame1 = ctk.CTkFrame(self.plot_container)
            frame1.pack(fill="both", expand=True, padx=8, pady=8)
            canvas1 = FigureCanvasTkAgg(fig1, master=frame1)
            canvas1.draw()
            canvas1.get_tk_widget().pack(fill="both", expand=True)
        except Exception as e:
            print("Error plotting histogram:", e)

        if "price" in df.columns:
            try:
                prices = pd.to_numeric(df["price"], errors="coerce")
                valid = df["distance_km"].notna() & prices.notna()
                fig2, ax2 = plt.subplots(figsize=(8, 4))
                if valid.sum() == 0:
                    ax2.text(0.5, 0.5, "No price vs distance data available", ha="center")
                else:
                    ax2.scatter(df.loc[valid, "distance_km"], prices[valid], alpha=0.4, s=8)
                ax2.set_title(f"Price vs Distance{title_add}")
                ax2.set_xlabel("Distance (km)")
                ax2.set_ylabel("Price")
                fig2.tight_layout()
                frame2 = ctk.CTkFrame(self.plot_container)
                frame2.pack(fill="both", expand=True, padx=8, pady=8)
                canvas2 = FigureCanvasTkAgg(fig2, master=frame2)
                canvas2.draw()
                canvas2.get_tk_widget().pack(fill="both", expand=True)
            except Exception as e:
                print("Error plotting price vs distance:", e)

        try:
            topd = df["destination_city"].value_counts().head(10)
            fig3, ax3 = plt.subplots(figsize=(8, 3.5))
            if topd.empty:
                ax3.text(0.5, 0.5, "No destination data available", ha="center")
            else:
                ax3.bar(topd.index.astype(str), topd.values)
                ax3.set_xticklabels(topd.index.astype(str), rotation=45, ha="right")
            ax3.set_title(f"Top destinations{title_add}")
            ax3.set_ylabel("Flights")
            fig3.tight_layout()
            frame3 = ctk.CTkFrame(self.plot_container)
            frame3.pack(fill="both", expand=True, padx=8, pady=8)
            canvas3 = FigureCanvasTkAgg(fig3, master=frame3)
            canvas3.draw()
            canvas3.get_tk_widget().pack(fill="both", expand=True)
        except Exception as e:
            print("Error plotting top destinations:", e)

    def recompute_caches(self):
        ok = messagebox.askyesno("Recompute caches",
                                 "This will (re)fetch coordinates for missing cities and recompute pairwise distances. Continue?")
        if not ok:
            return
        self.analytics.ensure_city_coords(verbose=True)
        self.analytics.compute_pair_distances(verbose=True)
        self.analytics.enrich_dataframe()
        self.df = self.analytics.df
        messagebox.showinfo("Done", "Caches refreshed and dataframe enriched.")
        self.show_summary(global_view=True)

    def export_csv(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile=os.path.basename(output_csv_default),
                                            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        outpath = self.analytics.export_enriched_csv(path)
        messagebox.showinfo("Exported", f"Enriched CSV saved to: {outpath}")

def main():
    analytics = AirlineAnalytics(csv_path)
    analytics.load_data()
    analytics.ensure_city_coords(verbose=True)
    analytics.compute_pair_distances(verbose=True)
    analytics.enrich_dataframe()
    root = ctk.CTk()
    gui = AirlineGUI(root, analytics)
    root.mainloop()

if __name__ == "__main__":
    main()
