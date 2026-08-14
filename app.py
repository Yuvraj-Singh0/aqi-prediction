"""
app.py - AQI Prediction System (Complete All-in-One)
Serves both the frontend and backend from a single Python file.
"""
import json, logging, os, pickle, sqlite3, threading, warnings
import urllib.parse, urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = "processed"
MODEL_DIR     = "models"
DB_PATH       = "predictions.db"
DATA_PATH     = os.path.join(PROCESSED_DIR, "processed_data.pkl")
SCALER_PATH   = os.path.join(PROCESSED_DIR, "scaler.pkl")
MODEL_PATH    = os.path.join(MODEL_DIR,     "model.pkl")
POLLUTANTS    = ["PM2_5","PM10","NO2","SO2","CO","O3"]
TARGET        = "AQI"
RANDOM_SEED   = 42
N_ROWS        = 8000

# ── Global state ──────────────────────────────────────────────────────────────
_model, _scaler, _feature_cols = None, None, []
pipeline_status = {"step":"idle","message":"Click Connect to start","progress":0,"error":None}

# ── EPA Data ──────────────────────────────────────────────────────────────────
EPA_BREAKPOINTS = {
    "PM2_5":[(0.0,12.0,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),(55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,500.4,301,500)],
    "PM10": [(0,54,0,50),(55,154,51,100),(155,254,101,150),(255,354,151,200),(355,424,201,300),(425,604,301,500)],
    "NO2":  [(0,53,0,50),(54,100,51,100),(101,360,101,150),(361,649,151,200),(650,1249,201,300),(1250,2049,301,500)],
    "SO2":  [(0,35,0,50),(36,75,51,100),(76,185,101,150),(186,304,151,200),(305,604,201,300),(605,1004,301,500)],
    "CO":   [(0.0,4.4,0,50),(4.5,9.4,51,100),(9.5,12.4,101,150),(12.5,15.4,151,200),(15.5,30.4,201,300),(30.5,50.4,301,500)],
    "O3":   [(0,54,0,50),(55,70,51,100),(71,85,101,150),(86,105,151,200),(106,200,201,300)],
}
AQI_CATEGORIES = [(0,50,"Good","#00e400"),(51,100,"Moderate","#ffff00"),(101,150,"Unhealthy for Sensitive Groups","#ff7e00"),(151,200,"Unhealthy","#ff0000"),(201,300,"Very Unhealthy","#8f3f97"),(301,500,"Hazardous","#7e0023")]
POLLUTANT_BOUNDS = {"PM2_5":(0,500),"PM10":(0,604),"NO2":(0,2049),"SO2":(0,1004),"CO":(0,50.4),"O3":(0,200)}

# ── Helpers ───────────────────────────────────────────────────────────────────
def sub_aqi(c, bps):
    for lo,hi,ilo,ihi in bps:
        if lo<=c<=hi: return ((ihi-ilo)/(hi-lo))*(c-lo)+ilo
    return 500.0

def epa_aqi(pol):
    vals = {p:sub_aqi(float(pol.get(p,0)),EPA_BREAKPOINTS[p]) for p in EPA_BREAKPOINTS}
    v = round(max(vals.values()),1)
    for lo,hi,label,colour in AQI_CATEGORIES:
        if lo<=v<=hi: return v,label,colour
    return v,"Hazardous","#7e0023"

def feature_vec(pol):
    now = datetime.now()
    h,dow,mon = now.hour,now.weekday(),now.month
    b = {**{p:pol.get(p,0) for p in POLLUTANTS},
         "hour":h,"day_of_week":dow,"month":mon,"is_weekend":int(dow>=5),
         "hour_sin":np.sin(2*np.pi*h/24),"hour_cos":np.cos(2*np.pi*h/24),
         "month_sin":np.sin(2*np.pi*mon/12),"month_cos":np.cos(2*np.pi*mon/12)}
    for p in POLLUTANTS:
        v=pol.get(p,0)
        b.update({f"{p}_lag1":v,f"{p}_lag24":v,f"{p}_roll6_mean":v,f"{p}_roll24_mean":v,f"{p}_roll6_std":0.0})
    b["PM_ratio"]       = b["PM2_5"]/b["PM10"] if b["PM10"] else 0
    b["NO2_O3_product"] = b["NO2"]*b["O3"]
    return np.array([b.get(c,0.0) for c in _feature_cols]).reshape(1,-1)

# ── Pipeline ──────────────────────────────────────────────────────────────────
def run_pipeline():
    global _model,_scaler,_feature_cols
    try:
        pipeline_status.update({"step":"preprocessing","message":"Generating 8,000 air quality records...","progress":5})
        rng=np.random.default_rng(RANDOM_SEED)
        dates=pd.date_range(start="2020-01-01",periods=N_ROWS,freq="h")
        hour=dates.hour.to_numpy(); month=dates.month.to_numpy()
        rush=1+0.4*np.exp(-((hour-8)**2)/8)+0.35*np.exp(-((hour-18)**2)/8)
        seas=1+0.3*np.cos((month-1)*2*np.pi/12+np.pi)
        base=rush*seas
        df=pd.DataFrame({"timestamp":dates})
        df["PM2_5"]=np.clip(rng.lognormal(2.5,0.6,N_ROWS)*base+rng.normal(0,1,N_ROWS),0.1,500)
        df["PM10"] =np.clip(df["PM2_5"]*rng.uniform(1.5,2.5,N_ROWS)+rng.normal(0,5,N_ROWS),1,600)
        df["NO2"]  =np.clip(rng.lognormal(3.2,0.7,N_ROWS)*base+rng.normal(0,2,N_ROWS),1,2049)
        df["SO2"]  =np.clip(rng.lognormal(2.0,0.8,N_ROWS)*base*0.5+rng.normal(0,1,N_ROWS),0,1004)
        df["CO"]   =np.clip(rng.lognormal(0.8,0.5,N_ROWS)*base+rng.normal(0,0.2,N_ROWS),0.1,50)
        df["O3"]   =np.clip(rng.lognormal(3.0,0.5,N_ROWS)*(1+0.2*np.sin(hour*np.pi/12))+rng.normal(0,3,N_ROWS),0,200)
        for col in POLLUTANTS:
            mask=rng.random(N_ROWS)<0.03; df.loc[mask,col]=np.nan
        pipeline_status.update({"message":"Cleaning and engineering features...","progress":20})
        df[POLLUTANTS]=df[POLLUTANTS].ffill(limit=2).bfill(limit=2)
        for col in POLLUTANTS:
            df[col]=df[col].clip(upper=df[col].quantile(0.995)).fillna(df[col].median())
        df["timestamp"]=pd.to_datetime(df["timestamp"]); df.set_index("timestamp",inplace=True)
        df["hour"]=df.index.hour; df["day_of_week"]=df.index.dayofweek; df["month"]=df.index.month
        df["is_weekend"]=df["day_of_week"].isin([5,6]).astype(int)
        df["hour_sin"]=np.sin(2*np.pi*df["hour"]/24); df["hour_cos"]=np.cos(2*np.pi*df["hour"]/24)
        df["month_sin"]=np.sin(2*np.pi*df["month"]/12); df["month_cos"]=np.cos(2*np.pi*df["month"]/12)
        for col in POLLUTANTS:
            df[f"{col}_lag1"]=df[col].shift(1); df[f"{col}_lag24"]=df[col].shift(24)
            df[f"{col}_roll6_mean"]=df[col].rolling(6,min_periods=1).mean()
            df[f"{col}_roll24_mean"]=df[col].rolling(24,min_periods=1).mean()
            df[f"{col}_roll6_std"]=df[col].rolling(6,min_periods=1).std().fillna(0)
        df["PM_ratio"]=(df["PM2_5"]/df["PM10"].replace(0,np.nan)).fillna(0)
        df["NO2_O3_product"]=df["NO2"]*df["O3"]
        df[TARGET]=df.apply(lambda r: max([sub_aqi(float(r.get(p,0)),EPA_BREAKPOINTS[p]) for p in EPA_BREAKPOINTS if pd.notna(r.get(p))],default=np.nan),axis=1)
        df.dropna(inplace=True)
        fcols=[c for c in df.columns if c!=TARGET]
        scaler=StandardScaler(); df[fcols]=scaler.fit_transform(df[fcols])
        os.makedirs(PROCESSED_DIR,exist_ok=True)
        with open(DATA_PATH,"wb") as f: pickle.dump(df,f)
        with open(SCALER_PATH,"wb") as f: pickle.dump(scaler,f)
        pipeline_status.update({"message":"Training Random Forest + XGBoost ensemble...","progress":45})
        X=df[fcols]; y=df[TARGET]
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=RANDOM_SEED)
        rf=RandomForestRegressor(n_estimators=100,max_depth=15,min_samples_split=5,random_state=RANDOM_SEED,n_jobs=-1)
        xgb=XGBRegressor(n_estimators=100,max_depth=5,learning_rate=0.1,subsample=0.8,colsample_bytree=0.8,random_state=RANDOM_SEED,n_jobs=-1,verbosity=0)
        ens=StackingRegressor(estimators=[("rf",rf),("xgb",xgb)],final_estimator=Ridge(),cv=3,n_jobs=-1)
        ens.fit(Xtr,ytr)
        pipeline_status.update({"message":"Evaluating model performance...","progress":88})
        yp=ens.predict(Xte)
        logger.info("MAE=%.3f RMSE=%.3f R2=%.4f",mean_absolute_error(yte,yp),np.sqrt(mean_squared_error(yte,yp)),r2_score(yte,yp))
        os.makedirs(MODEL_DIR,exist_ok=True)
        with open(MODEL_PATH,"wb") as f: pickle.dump(ens,f)
        _model=ens; _scaler=scaler; _feature_cols=fcols
        pipeline_status.update({"step":"done","message":"Model ready! API connected.","progress":100})
        logger.info("✅ Pipeline complete")
    except Exception as e:
        pipeline_status.update({"step":"error","message":str(e),"progress":0,"error":str(e)})
        logger.exception("Pipeline failed")

def load_model():
    global _model,_scaler,_feature_cols
    with open(MODEL_PATH,"rb") as f: _model=pickle.load(f)
    with open(SCALER_PATH,"rb") as f: _scaler=pickle.load(f)
    with open(DATA_PATH,"rb") as f: df=pickle.load(f)
    _feature_cols=[c for c in df.columns if c!=TARGET]
    pipeline_status.update({"step":"done","message":"Model ready! API connected.","progress":100})
    logger.info("✅ Model loaded")

# ── Flask App ─────────────────────────────────────────────────────────────────
app=Flask(__name__)
CORS(app,resources={r"/*":{"origins":"*","methods":["GET","POST","DELETE","OPTIONS"],"allow_headers":["Content-Type"]}})

def init_db():
    with sqlite3.connect(DB_PATH) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS predictions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,timestamp TEXT NOT NULL,
            pm2_5 REAL,pm10 REAL,no2 REAL,so2 REAL,co REAL,o3 REAL,
            aqi_epa REAL,aqi_model REAL,category TEXT,colour TEXT)""")
        c.commit()

@app.before_request
def preflight():
    if request.method=="OPTIONS": return "",204

@app.route("/")
def index():
    return send_file("index.html")

@app.route("/health")
def health():
    return jsonify({"status":"ok","model_loaded":_model is not None,"pipeline_step":pipeline_status["step"],"pipeline_msg":pipeline_status["message"],"progress":pipeline_status["progress"]}),200

@app.route("/start-pipeline",methods=["POST"])
def start_pipeline():
    if pipeline_status["step"] in ("preprocessing","training","loading"): return jsonify({"message":"Already running"}),200
    if pipeline_status["step"]=="done" and _model: return jsonify({"message":"Already loaded"}),200
    threading.Thread(target=run_pipeline,daemon=True).start()
    return jsonify({"message":"Started"}),200

@app.route("/pipeline-status")
def get_status():
    return jsonify(pipeline_status),200

@app.route("/predict",methods=["POST"])
def predict():
    if not _model: return jsonify({"error":"Model not ready. Click Connect first."}),503
    data=request.get_json(silent=True) or {}
    errors,parsed={},{}
    for pol in POLLUTANTS:
        if pol not in data: errors[pol]="Missing"; continue
        try: val=float(data[pol])
        except: errors[pol]="Invalid"; continue
        lo,hi=POLLUTANT_BOUNDS[pol]
        if not(lo<=val<=hi): errors[pol]=f"Must be {lo}-{hi}"
        else: parsed[pol]=val
    if errors: return jsonify({"error":"Validation failed","details":errors}),422
    av,cat,col=epa_aqi(parsed)
    try: am=round(float(np.clip(_model.predict(_scaler.transform(feature_vec(parsed)))[0],0,500)),1)
    except: am=av
    try:
        with sqlite3.connect(DB_PATH) as c:
            c.execute("INSERT INTO predictions(timestamp,pm2_5,pm10,no2,so2,co,o3,aqi_epa,aqi_model,category,colour) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                     (datetime.utcnow().isoformat(),parsed.get("PM2_5"),parsed.get("PM10"),parsed.get("NO2"),parsed.get("SO2"),parsed.get("CO"),parsed.get("O3"),av,am,cat,col))
            c.commit()
    except: pass
    return jsonify({"aqi_epa":av,"aqi_model":am,"category":cat,"colour":col,"pollutants":parsed}),200

@app.route("/history")
def history():
    try:
        with sqlite3.connect(DB_PATH) as c:
            c.row_factory=sqlite3.Row
            rows=c.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 20").fetchall()
        return jsonify({"count":len(rows),"records":[dict(r) for r in rows]}),200
    except Exception as e: return jsonify({"error":str(e)}),500

@app.route("/clear-history",methods=["DELETE","POST"])
def clear_history():
    try:
        with sqlite3.connect(DB_PATH) as c:
            cur=c.cursor(); cur.execute("SELECT COUNT(*) FROM predictions"); n=cur.fetchone()[0]
            cur.execute("DELETE FROM predictions"); c.commit()
        return jsonify({"status":"cleared","records_deleted":n}),200
    except Exception as e: return jsonify({"error":str(e)}),500

@app.route("/location/<city>")
def location(city):
    CITIES={"delhi":("Delhi","IN",28.6139,77.2090),"new delhi":("New Delhi","IN",28.6139,77.2090),
            "mumbai":("Mumbai","IN",19.0760,72.8777),"kolkata":("Kolkata","IN",22.5726,88.3639),
            "chennai":("Chennai","IN",13.0827,80.2707),"bangalore":("Bangalore","IN",12.9716,77.5946),
            "bengaluru":("Bengaluru","IN",12.9716,77.5946),"hyderabad":("Hyderabad","IN",17.3850,78.4867),
            "pune":("Pune","IN",18.5204,73.8567),"ahmedabad":("Ahmedabad","IN",23.0225,72.5714),
            "jaipur":("Jaipur","IN",26.9124,75.7873),"lucknow":("Lucknow","IN",26.8467,80.9462),
            "gurugram":("Gurugram","IN",28.4595,77.0266),"noida":("Noida","IN",28.5355,77.3910),
            "london":("London","GB",51.5074,-0.1278),"new york":("New York","US",40.7128,-74.0060),
            "nyc":("NYC","US",40.7128,-74.0060),"tokyo":("Tokyo","JP",35.6762,139.6503),
            "beijing":("Beijing","CN",39.9042,116.4074),"paris":("Paris","FR",48.8566,2.3522),
            "dubai":("Dubai","AE",25.2048,55.2708),"singapore":("Singapore","SG",1.3521,103.8198),
            "sydney":("Sydney","AU",-33.8688,151.2093),"toronto":("Toronto","CA",43.6532,-79.3832),
            "lahore":("Lahore","PK",31.5204,74.3587),"dhaka":("Dhaka","BD",23.8103,90.4125)}
    REGION_POLL={"IN":{"pm25":(40,120),"pm10":(80,200),"no2":(30,80),"so2":(10,40),"co":(500,1500),"o3":(30,80)},
                 "CN":{"pm25":(50,150),"pm10":(90,220),"no2":(35,90),"so2":(15,50),"co":(600,1800),"o3":(40,90)},
                 "US":{"pm25":(5,25),"pm10":(15,60),"no2":(15,50),"so2":(2,15),"co":(200,600),"o3":(30,70)},
                 "XX":{"pm25":(10,60),"pm10":(25,100),"no2":(15,60),"so2":(5,30),"co":(300,1000),"o3":(25,70)}}
    k=city.lower().strip(); info=CITIES.get(k)
    if not info: info=("Unknown","XX",20.5937,78.9629)
    name,ctry,lat,lon=info
    reg=REGION_POLL.get(ctry,REGION_POLL["XX"])
    def rv(lo,hi): return round(float(np.random.uniform(lo,hi)),1)
    return jsonify({"city":name,"country":ctry,"latitude":lat,"longitude":lon,
                    "PM2_5":rv(*reg["pm25"]),"PM10":rv(*reg["pm10"]),
                    "NO2":rv(*reg["no2"]),"SO2":rv(*reg["so2"]),
                    "CO":round(rv(*reg["co"])/1150,2),"O3":round(rv(*reg["o3"])/1.96,1)}),200

if __name__=="__main__":
    init_db()
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(DATA_PATH):
        try: load_model()
        except Exception as e:
            logger.error("Load failed: %s",e)
            threading.Thread(target=run_pipeline,daemon=True).start()
    else:
        threading.Thread(target=run_pipeline,daemon=True).start()
    port=int(os.environ.get("PORT",5000))
    print(f"\n{'='*50}\n  AQI System running → http://127.0.0.1:{port}\n{'='*50}\n")
    app.run(host="0.0.0.0",port=port,debug=False,threaded=True)
