
# FIXED HEADER INSERTED
import os
from kiteconnect import KiteConnect
def get_kite():
    api_key = os.getenv("KITE_API_KEY")
    access_token = os.getenv("KITE_ACCESS_TOKEN")
    if not api_key or not access_token:
        raise Exception("Missing KITE_API_KEY or KITE_ACCESS_TOKEN in .env")
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite

kite = get_kite()
HIST_CACHE = {}
HIST_CACHE_TIME = {}



# REAL_TIME_ALERTS.py  ✅ FINAL FIX

import os
import sys

USERNAME = sys.argv[2] if len(sys.argv) > 2 else "default_user"
SAFE_USER = USERNAME.strip().replace(" ", "_")

IS_RENDER = bool(os.getenv("RENDER"))

# 🔥 IMPORTANT FIX — align with backend/app/data
BACKEND_ROOT = os.path.dirname(os.path.abspath(__file__))

if IS_RENDER:
    INTERIM_FOLDER = "/data"
else:
    INTERIM_FOLDER = os.path.join(BACKEND_ROOT, "app", "data")

os.makedirs(INTERIM_FOLDER, exist_ok=True)

SIG2_PATH  = os.path.join(INTERIM_FOLDER, f"{SAFE_USER}_sig_2mins.csv")
SIG15_PATH = os.path.join(INTERIM_FOLDER, f"{SAFE_USER}_sig_15mins.csv")


os.makedirs(INTERIM_FOLDER, exist_ok=True)

print("📍 ENV =", "RENDER" if IS_RENDER else "LOCAL")
print("📍 INTERIM_FOLDER =", INTERIM_FOLDER)
print("📍 SIG2_PATH =", SIG2_PATH)
print("📍 SIG15_PATH =", SIG15_PATH)

import numpy as np # linear algebra
import pandas as pd # pandas for dataframe based data processing and CSV file I/O
import requests # for http requests
from bs4 import BeautifulSoup # for html parsing and scraping
import bs4


import requests
import re
import pandas as pd
import json
# api-endpoint


import datetime
from datetime import timedelta


header = {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36",
         "accept-language": "en-US,en;q=0.9", "accept-encoding": "gzip, deflate, br"}


# ========== NEW CELL ==========


import pandas as pd
import os
import numpy as np




from datetime import datetime, date, timedelta
import dateutil.relativedelta as rel
import warnings
warnings.WarningMessage
warnings.filterwarnings("ignore")
import ta
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.info("test")
import sys
import yfinance as yf
from yfinance import Ticker as tick
#data = yf.download(tickers='SBIN.NS', period='700d', interval='1h')

period_start=1
period_start_value=0
kick_start_minute_job=0



import os
import pandas as pd
import numpy as np


# -----------------------------
# Always create signal files (avoid Errno 2 on new users)
# -----------------------------
def _init_signal_files(sig2_path: str, sig15_path: str):
    base_cols = ["Date", "open_price", "high_price", "low_price", "close_price", "vol", "script"]
    for p in [sig2_path, sig15_path]:
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            if not os.path.exists(p):
                pd.DataFrame(columns=base_cols).to_csv(p, index=False)
        except Exception as e:
            print(f"❌ Could not init signal file {p}: {e}")

def _safe_write_csv(df: pd.DataFrame, path: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if df is None:
            _init_signal_files(path, path)
            return
        df.to_csv(path, index=False)
    except Exception as e:
        print(f"❌ CSV write failed for {path}: {e}")




def _make_unique_columns(cols):
    """
    Pandas can carry duplicate column names. This makes them unique by appending __dupN.
    Example: ['script','script'] -> ['script','script__dup1']
    """
    seen = {}
    out = []
    for c in list(cols):
        c = str(c)
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}__dup{seen[c]}")
    return out

def _coalesce_prefixed_dups(df: pd.DataFrame, base_name: str) -> pd.DataFrame:
    """
    If we created columns like script, script__dup1, script__dup2,
    coalesce them into one 'script' by taking first non-null across them.
    """
    cols = [c for c in df.columns if c == base_name or c.startswith(base_name + "__dup")]
    if len(cols) <= 1:
        return df

    s = df[cols[0]]
    for c in cols[1:]:
        s = s.combine_first(df[c])

    # Drop ALL those cols and add back one clean base col
    df = df.drop(columns=cols, errors="ignore")
    df[base_name] = s
    return df

def normalize_sr_df(df: pd.DataFrame, *, fallback_script: str = None) -> pd.DataFrame:
    """
    Normalizes SR dataframe before Resistance/F6:
    - makes duplicate column names unique
    - canonicalizes key columns (Date/script/ohlc/vol)
    - merges Script + script into a single script
    - removes duplicates of script/date/etc
    """
    if df is None or len(df) == 0:
        return df

    df = df.copy()

    # 1) Make duplicate column names unique (critical!)
    df.columns = _make_unique_columns(df.columns)

    # 2) Build case-insensitive rename map for common columns
    lower_map = {c.lower(): c for c in df.columns}

    def _rename_if_present(src_lower, dst):
        if src_lower in lower_map and dst not in df.columns:
            df.rename(columns={lower_map[src_lower]: dst}, inplace=True)

    # Date
    _rename_if_present("date", "Date")
    _rename_if_present("datetime", "Date")
    _rename_if_present("timestamp", "Date")

    # Script
    _rename_if_present("script", "script")
    _rename_if_present("symbol", "script")
    _rename_if_present("tradingsymbol", "script")
    _rename_if_present("scriptname", "script")
    _rename_if_present("Script".lower(), "script")  # handles Script/ SCRIPT etc.

    # OHLC + vol
    _rename_if_present("open", "open_price")
    _rename_if_present("high", "high_price")
    _rename_if_present("low", "low_price")
    _rename_if_present("close", "close_price")
    _rename_if_present("volume", "vol")

    # If your original feed already uses *_price and vol, this does nothing.

    # 3) Coalesce duplicates created by rename / existing columns
    #    (example: you had 'Script' and 'script' -> after rename you may have multiple script columns)
    for base in ["script", "Date", "open_price", "high_price", "low_price", "close_price", "vol"]:
        df = _coalesce_prefixed_dups(df, base)

    # 4) Guarantee script exists
    if "script" not in df.columns:
        if fallback_script is not None:
            df["script"] = fallback_script
        else:
            # create empty but present; downstream may fail without fallback
            df["script"] = ""

    # If script exists but is blank/NaN, fill from fallback
    if fallback_script is not None:
        df["script"] = df["script"].replace("", np.nan).fillna(fallback_script)

    # 5) Ensure Date is datetime if present
    if "Date" in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["Date"])
        except Exception:
            pass

    # 6) FINAL: remove any *exact* duplicated columns (after all merges)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    return df




# ========== NEW CELL ==========

def _canon_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Make column names stable across OS/Render (strip spaces, normalize script/date)."""
    if df is None or df.empty:
        return df

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename = {}
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc == "script":
            rename[c] = "script"
        elif lc == "date":
            rename[c] = "Date"

    if rename:
        df.rename(columns=rename, inplace=True)

    # If still missing 'script' but has common variants
    if "script" not in df.columns:
        for alt in ["Script", "SCRIPT", "tradingsymbol", "TradingSymbol"]:
            if alt in df.columns:
                df.rename(columns={alt: "script"}, inplace=True)
                break

    return df

def date_clean_function(infile="intraday_nsedata_15min.csv"):

    ##S1: Read file
    df = pd.read_csv(infile)
    df['Date'] = pd.to_datetime(df['Date'])

    ##S2: Clean the timezone by adding the 5h30mins
    start_time = pd.to_datetime('09:15:00').time()
    end_time = pd.to_datetime('15:30:00').time()

    # Add 5 hours and 30 minutes to dates outside the range
    df['Date'] = df['Date'].apply(lambda x: x + pd.Timedelta(hours=5, minutes=30)
                                                if (x.time() < start_time or x.time() > end_time) else x)


    ##S3: Keep only those rows with 15min gaps...
    df['Date'] = pd.to_datetime(df['Date'])

    if "script" in df.columns:
        SCRIPT = "script"
    elif "Script" in df.columns:
        SCRIPT = "Script"

    df = df[df['Date'].dt.minute % 15 == 0].drop_duplicates(['Date',SCRIPT])


    df.to_csv(infile, index=False)

    return df


## df1 = date_clean_function(infile="intraday_nsedata_15min.csv")

# ========== NEW CELL ==========

minute_df=pd.DataFrame()

# ========== NEW CELL ==========

def complete_data_prep_intraday(script_to_check = "VOLTAS", infile = "Intraday_nsedata.csv" , infile_min = 'intraday_nsedata_15min.csv'):

    """
    Logic:
     - Read in the infile that contains atleast the list of scripts. So if this file needs to prepare from scratch, only put the list of script in script column and any date to kick-start this macro
     -

    """

    kick_start_minute_job = 0
    period_end_value = 0


    #Step1: Read Input file, filter data for a script, check whether the last price stored in the dataset is the latest one.
    # If not, then contiunue with recording the new data process....

    read_datafile = pd.read_csv(infile)
    read_datafile["Date"] = pd.to_datetime(read_datafile["Date"])
    read_datafile_filter = read_datafile[(read_datafile.Script == script_to_check)]
    read_datafile_filter = read_datafile_filter[read_datafile_filter.Date == np.max(read_datafile_filter['Date'])]
    close_price_to_check = read_datafile_filter.iloc[0,:]["close_price"]

    x = ns.get_quote(script_to_check)
    last_price = np.float(str(x['data'][0]['lastPrice']).replace(',',''))

    #Step2: If the price is updated at NSE website, continue with following steps:
    if close_price_to_check != last_price:
        print("As new Data is present, it's proceeding with collecting the new data !!!")

        run_next_steps = 1

        ##### Step2a: Read nse_inputdata.csv file that has 2 mins granularity data, arrange the file in ascensing order by script & Date ####
        daily_data_file = pd.read_csv(infile)
        symbol_list = list(daily_data_file.Script.drop_duplicates())

        daily_data_file["Date"] = pd.to_datetime(daily_data_file["Date"])

        #ensure that there are no duplicate observations present in input file... if there are any please remove them
        daily_data_file = daily_data_file.drop_duplicates(['Date','Script']).sort_values(by= ["Script","Date"])

        st_data = pd.DataFrame()

        count =0
        period_end_value = 0

        #Step2b - For each_Script, we shall be downloading data from website...
        for i in symbol_list :
            try:
                if count ==0:
                    print("Data downloading Loop has started for script {}!!!".format(i))
                x = ns.get_quote(i)   ### getting latestprice for the script = i, data comes in dictionary format
                y=pd.DataFrame(x['data'])   ### storing data inform of DataFrame
                y['lastUpdateTime'] = pd.to_datetime(x['lastUpdateTime'])
                y['minute'] = y['lastUpdateTime'].dt.minute
                y['hour'] = y['lastUpdateTime'].dt.hour

                ## Define 15 mins interval agg_period
                y["agg_period"] = np.where(y["minute"] == 0,y["hour"].astype('str') + ":00",
                                  np.where((y["minute"] > 0) & (y["minute"] <=15), y["hour"].astype('str') + ":15",
                                  np.where((y["minute"] >15) & (y["minute"] <= 30), y["hour"].astype('str') + ":30",
                                  np.where((y["minute"] >30) & (y["minute"] <= 45), y["hour"].astype('str') + ":45",(y["hour"] +1).astype('str') + ":00"))))

                #Loop counter
                if count ==0:
                    logging.info("Data Collection started ....")

                #Get the last row of the dataframe i.e. last 2 min candle
                df = daily_data_file[daily_data_file.Script == i].tail(1)

                #If the last candle for this script is present in the data & current time-stamp is not same as last recorded timestamp then execute following condition --
                #cond1: This is important because let say if no data is present for this script then following processing is not required
                #cond2: To avoid duplicate records because these tend to collect inaccurate data..

                if (df.Date.count() > 0):

                    y["date_new"] = datetime.strftime(pd.to_datetime(y['lastUpdateTime'][0]),'%m/%d/%Y')
                    y['lastPrice'] = y['lastPrice'].apply(lambda x: np.float(str(x).replace(',','').replace('-','0')))


                st_data = st_data.append(y[['agg_period','symbol','lastUpdateTime','totalTradedVolume','totalSellQuantity','totalBuyQuantity','averagePrice','lastPrice']])
                print(i)
                count = count + 1


                if (np.round(count/len(symbol_list)*100,0) == 25) or \
                    (np.round(count/len(symbol_list)*100,0) == 50) or \
                    (np.round(count/len(symbol_list)*100,0) == 75) or \
                    (np.round(count/len(symbol_list)*100,0) == 100):
                        logging.info(str(np.round(count/len(symbol_list)*100,0)))
            except:
                continue

        temp = pd.read_csv("temp.csv")

        if temp.count()[0] > 0:
            if (str(y.date_new.iloc[0]) +" " + y["agg_period"].iloc[0]) != str(temp["start"].tail(1).iloc[0]):
                kick_start_minute_job = 1
                period_end_value = str(temp["start"].tail(1).iloc[0])

        interim_temp = temp[temp.start.isnull()]
        interim_temp["current_time"] = y['lastUpdateTime']
        interim_temp["agg_period"] = y['agg_period']
        interim_temp["start"] = str(y.date_new.iloc[0]) +" " + y["agg_period"].iloc[0]
        temp = temp.append(interim_temp)
        temp.to_csv("temp.csv", index = False)

        finaldataread = st_data.reset_index()
        finaldataread = finaldataread.rename({'symbol':'Script','Symbol':'Script','lastUpdateTime':'Date','averagePrice':'VWAP', 'lastPrice':'close_price','totalTradedVolume':'vol','open':'open_price','dayHigh':'high_price','dayLow':'low_price'},axis=1).sort_values(by = ["Script", "Date"], ascending = False)


        print ("kick_start_minute_job =" +  str(kick_start_minute_job))
        print ("period_end_value = " + str(period_end_value))


        masterdata_trend = daily_data_file.append(finaldataread)


        ####Clean unwanted observations if present--->
        masterdata_trend = masterdata_trend[masterdata_trend.close_price.isnull() == False]


        ###Format columns --->

        masterdata_trend["Date"] = pd.to_datetime(masterdata_trend["Date"])

        float_cols = ['VWAP', 'close_price','index']

        for i in float_cols:
            masterdata_trend[i] = masterdata_trend[i].apply(lambda x: np.float(str(x).replace(',','')))

        int_cols = ['totalBuyQuantity','totalSellQuantity', 'vol']

        for i in int_cols:
            masterdata_trend[i] = masterdata_trend[i].apply(lambda x: np.float(str(x).replace(',','').replace('-','0')))


        #### Output1 - Two Minutes Intraday data
        masterdata_trend = masterdata_trend.drop_duplicates(['Date','Script'])
        masterdata_trend.to_csv(infile,index=False)

        dic = dict()

        candle_cnt = pd.read_csv("Master_candle_counts.csv")
        candle_cnt['agg'] = candle_cnt['agg'].apply(lambda x: x.strip())

        minute_df = pd.read_csv(infile_min)
        minute_df['agg'] = minute_df.Date.apply(lambda x: str(x).split(' ')[1].strip()[:5])
        if "candle_no" in minute_df.columns:
            minute_df = minute_df.drop(['candle_no'],axis=1).merge(candle_cnt,on = ['agg'], how='left').drop_duplicates()
        else:
            minute_df = minute_df.merge(candle_cnt,on = ['agg'], how='left').drop_duplicates()
        minute_df.Date = pd.to_datetime(minute_df.Date)
        masterdata_trend.Date = pd.to_datetime(masterdata_trend.Date)


        if kick_start_minute_job == 1:
            print ("Minutes data updation started for " + str(period_end_value) + "......")
            #minute_df = pd.read_csv(infile_min)
            df = masterdata_trend[(masterdata_trend.Date >= pd.to_datetime(period_end_value) - pd.Timedelta(minutes = 15)) & (masterdata_trend.Date <= pd.to_datetime(period_end_value))].sort_values(by = ["Script","Date"])
            for each_script in symbol_list:
                try:
                    df_filter = df[df.Script == each_script]
                    dic["Date"] = str(df_filter.Date.dt.date.iloc[0]) + " " + str(df_filter.agg_period.iloc[0])
                    dic['agg'] = str(df_filter.agg_period.iloc[0]).strip()
                    dic['agg'] = "0"+dic['agg'] if dic['agg'][0] == '9' else dic['agg']
                    dic["Script"] = df_filter["Script"].iloc[0]
                    dic["VWAP"] = np.sum(df_filter["VWAP"] * df_filter["vol"])/np.sum(df_filter["vol"])
                    dic["close_price"] = df_filter["close_price"].iloc[df_filter.count()[0]-1]
                    dic["high_price"] = np.max(df_filter['close_price'])
                    dic["low_price"] = np.min(df_filter['close_price'])
                    dic["open_price"] = df_filter["close_price"].iloc[0]
                    dic["totalBuyQuantity"] = np.sum(df_filter["totalBuyQuantity"])
                    dic["totalSellQuantity"] = np.sum(df_filter["totalSellQuantity"])
                    dic["vol"] = np.max(df_filter["vol"])

                    dic["vol_orig"] = np.max(df_filter["vol"])

                    interim_minute_df = pd.DataFrame(dic,index = [0])
                    interim_minute_df['Date'] = pd.to_datetime(interim_minute_df['Date'])
                    interim_minute_df = interim_minute_df.merge(candle_cnt, on = ['agg'], how='left').drop_duplicates()

                    ### Manipulating the vol column ---
                    df_filter_15 = minute_df[(minute_df.Script == each_script)].tail(1)

                    date_15 = df_filter_15.Date.dt.date.iloc[0]
                    min_15 = df_filter_15.Date.dt.minute.iloc[0]
                    hour_15 = df_filter_15.Date.dt.hour.iloc[0]
                    candle_no_15 = df_filter_15.candle_no.iloc[0]


                    date_curr = interim_minute_df.Date.dt.date.iloc[0]
                    min_curr= interim_minute_df.Date.dt.minute.iloc[0]
                    hour_curr = interim_minute_df.Date.dt.hour.iloc[0]
                    candle_no_curr = interim_minute_df.candle_no.iloc[0]


                    #1. If the first candle of the day is recorded during the later part of the day i.e. initial few candles are missed then current vol is equally distributed among missing candles, candle1 starting at 9:30

                    if (date_curr != date_15) & (candle_no_curr > 1):
                        missing_candles = candle_no_curr
                        interim_minute_df['vol'] = (interim_minute_df['vol_orig'].iloc[0])/missing_candles
                        print ("cond1 executed")

                    #2. If the first candle of the day belongs to the time-slot 9:30 then do nothing..

                    #3. For the following sequential candles during the day, take difference w.r.t previous vol
                    missing_candles = candle_no_curr - candle_no_15
                    #print (" missing candles: {}".format(missing_candles))

                    if (date_curr == date_15) & (missing_candles == 1):
                        interim_minute_df['vol'] = (interim_minute_df['vol_orig'].iloc[0] - df_filter_15['vol_orig'].iloc[0])
                        print ("cond2 executed: If Part")
                    elif (date_curr == date_15) & (missing_candles > 1):
                        interim_minute_df['vol'] = (interim_minute_df['vol_orig'].iloc[0] - df_filter_15['vol_orig'].iloc[0])/missing_candles
                        print ("cond2 executed: ELIf Part")

                    minute_df = minute_df.append(interim_minute_df)

                except:
                    continue

            minute_df.drop_duplicates(['Date','Script']).to_csv(infile_min,index = False)
            no_obs = minute_df[minute_df.Script == script_to_check].drop_duplicates("Date").count()[0]
        else:
            no_obs=0


        print ("Step1 of DataPrep from NSE site is completed!")


    else:
        run_next_steps=0
        print ("run_next_steps: = " + str(run_next_steps))
        minute_df = pd.read_csv(infile_min)
        masterdata_trend = pd.read_csv(infile)
        no_obs = 0

    return run_next_steps, no_obs, minute_df,masterdata_trend,kick_start_minute_job,period_end_value


# ========== NEW CELL ==========




index_list = ['NIFTY 50',
 'NIFTY BANK',
 'INDIA VIX',
 'NIFTY MIDCAP 100',
 'NIFTY INFRA',
 'NIFTY REALTY',
 'NIFTY ENERGY',
 'NIFTY FMCG',
 'NIFTY MNC',
 'NIFTY PHARMA',
 'NIFTY PSE',
 'NIFTY PSU BANK',
 'NIFTY SERV SECTOR',
 'NIFTY IT',
 'NIFTY AUTO',
 'NIFTY MEDIA',
 'NIFTY METAL',
 'NIFTY COMMODITIES',
 'NIFTY CONSUMPTION',
 'NIFTY CPSE',
 'NIFTY FIN SERVICE',
 'NIFTY PVT BANK',
 'NIFTY SMLCAP 50']



def index_data_collection(  index_list,
                            kick_start_minute_job,
                            period_end_value,
                            infile = "Intraday_indexdata.csv",
                            infile_min = 'Intraday_indexdata_15min.csv',
                            script_to_check = "NIFTY 50"):

    dic = {}
    temp_df = pd.DataFrame()

    read_datafile = pd.read_csv(infile)
    read_datafile["Date"] = pd.to_datetime(read_datafile["Date"])
    read_datafile_filter = read_datafile[(read_datafile.Script == script_to_check)]
    read_datafile_filter = read_datafile_filter[read_datafile_filter.Date == np.max(read_datafile_filter['Date'])]
    read_datafile_filter = read_datafile_filter.sort_values(['Date'], ascending = False).reset_index()
    close_price_to_check = read_datafile_filter.iloc[0,:]["close_price"]

    x = nse.get_index_quote(script_to_check)
    last_price = np.float(str(x['lastPrice']).replace(',',''))

    if close_price_to_check != last_price:
        print("As new Data is present, it's proceeding with collecting the new data !!!")

        run_next_steps = 1

        ##### Step2a: Read nse_inputdata.csv file that has 2 mins granularity data, arrange the file in ascensing order by script & Date ####
        daily_data_file = pd.read_csv(infile)
        symbol_list = list(daily_data_file.Script.drop_duplicates())

        daily_data_file["Date"] = pd.to_datetime(daily_data_file["Date"])

        #ensure that there are no duplicate observations present in input file... if there are any please remove them
        daily_data_file = daily_data_file.drop_duplicates(['Date','Script']).sort_values(by= ["Script","Date"])

        st_data = pd.DataFrame()

        count =0


        for i in index_list :
            try:
                if count ==0:
                    print("Data downloading Loop has started !!!")

                x = x = nse.get_index_quote(i)   ### getting latestprice for the script = i, data comes in dictionary format
                y=pd.DataFrame(x,index=[0])   ### storing data inform of DataFrame
                y['lastUpdateTime'] = pd.to_datetime(datetime.now().strftime('%m/%d/%Y %H:%M:%S' ))
                y['minute'] = y['lastUpdateTime'].dt.minute
                y['hour'] = y['lastUpdateTime'].dt.hour

                ## Define 15 mins interval agg_period
                y["agg_period"] = np.where(y["minute"] == 0,y["hour"].astype('str') + ":00",
                                  np.where((y["minute"] > 0) & (y["minute"] <=15), y["hour"].astype('str') + ":15",
                                  np.where((y["minute"] >15) & (y["minute"] <= 30), y["hour"].astype('str') + ":30",
                                  np.where((y["minute"] >30) & (y["minute"] <= 45), y["hour"].astype('str') + ":45",(y["hour"] +1).astype('str') + ":00"))))

                y["date_new"] = datetime.strftime(pd.to_datetime(y['lastUpdateTime'][0]),'%m/%d/%Y')
                #Loop counter
                if count ==0:
                    logging.info("Data Collection started ....")

                #Get the last row of the dataframe i.e. last 2 min candle
                df = daily_data_file[daily_data_file.Script == i].tail(1)

                #If the last candle for this script is present in the data & current time-stamp is not same as last recorded timestamp then execute following condition --
                #cond1: This is important because let say if no data is present for this script then following processing is not required
                #cond2: To avoid duplicate records because these tend to collect inaccurate data..

                if (df.Date.count() > 0):

                    y['lastPrice'] = y['lastPrice'].apply(lambda x: np.float(str(x).replace(',','').replace('-','0')))


                st_data = st_data.append(y[['lastUpdateTime','date_new','agg_period','name','lastPrice']])
                print(i)
                count = count + 1


                if (np.round(count/len(symbol_list)*100,0) == 25) or \
                    (np.round(count/len(symbol_list)*100,0) == 50) or \
                    (np.round(count/len(symbol_list)*100,0) == 75) or \
                    (np.round(count/len(symbol_list)*100,0) == 100):
                        logging.info(str(np.round(count/len(symbol_list)*100,0)))
            except:
                continue

        st_data = st_data.rename({'lastUpdateTime':'Date',
                         'agg_period':'agg',
                         'name':'Script',
                         'lastPrice':'close_price',
                         'date_new':'Date_short'}, axis=1)

        daily_data_file = daily_data_file.append(st_data)
        daily_data_file = daily_data_file.drop_duplicates(['Script','Date'])

        #Output1 ---->
        daily_data_file.to_csv(infile, index=False)

        if kick_start_minute_job == 1:
            print ("Minutes data updation started for " + str(period_end_value) + "......")
            #minute_df = pd.read_csv(infile_min)

            print(period_end_value)

            df = daily_data_file[(daily_data_file.Date >= (pd.to_datetime(period_end_value) - pd.Timedelta(minutes = 15))) &
                                 (daily_data_file.Date <= pd.to_datetime(period_end_value))].sort_values(by = ["Script","Date"])
            print(df.head(5))

            for each_script in index_list:

                    df_filter = df[df.Script == each_script]
                    print ("count of observations in filtered data: " + str(df_filter.count()[0]))
                    df_filter = df_filter.reset_index()
                    dic["Date"] = str(df_filter.Date.dt.date.iloc[0]) + " " + str(df_filter['agg'].iloc[0])
                    dic['agg'] = str(df_filter['agg'].iloc[0]).strip()
                    dic['agg'] = "0"+dic['agg'] if dic['agg'][0] == '9' else dic['agg']
                    dic["Script"] = df_filter["Script"].iloc[0]
                    dic["close_price"] = df_filter["close_price"].iloc[df_filter.count()[0]-1]
                    dic["high_price"] = np.max(df_filter['close_price'])
                    dic["low_price"] = np.min(df_filter['close_price'])
                    dic["open_price"] = df_filter["close_price"].iloc[0]
                    interim_minute_df = pd.DataFrame(dic,index = [0])
                    interim_minute_df['Date'] = pd.to_datetime(interim_minute_df['Date'])

                    temp_df = temp_df.append(interim_minute_df)

        print (temp_df)
        #### Read 15mins file
        minutes_df = pd.read_csv(infile_min)
        print('infile_min read successfully')

        minutes_df = minutes_df.append(temp_df)

        ## Output2 --->
        minutes_df.drop_duplicates(['Date','Script']).to_csv(infile_min,index = False)

    return

# ========== NEW CELL ==========

## Add Trend and output in the new file ...
def last_1_days(data, columns, last_day, ASC = True):

    data['Date'] = pd.to_datetime(data['Date'])

    if "script" in list(data.columns):
        script_col = "script"
    else:
        script_col = "Script"

    data = data.sort_values([script_col, 'Date'], ascending = ASC)
    for col in columns:
        for i in range(1, last_day + 1):
            data[f'{col}_{i}'] = data.groupby(script_col)[col].shift(i)

    data = data.fillna(0)

    return data

## Generic column shifting function
def columns_shift(df = pd.DataFrame(),lst_input = ['Script','trend_values'], shift_count = 1):

    lst = ['Script']
    lst.extend(lst_input)
    for i in lst:
        for j in range(shift_count):
            df[i+"_shift"+str(j+1)] = df[i].shift(j+1)

            if i != 'Script':
                df[i+"_shift"+str(j+1)] = np.where((df["Script_shift"+str(j+1)] == df["Script"]),df[i+"_shift"+str(j+1)],df[i])

    return df


# ========== NEW CELL ==========


# ========== NEW CELL ==========

import numpy as np
import pandas as pd

def trend_reversal_pattern2(
    infile: str | None = "osc_all.csv",
    path: str | None = None,               # kept for signature compatibility (unused)
    xlsx_file_name: str | None = None,     # kept for signature compatibility (unused)
    no_candles_for_performance: int = 20,  # kept for signature compatibility (unused)
    df: pd.DataFrame | None = None,
    *,
    pos_limit: int = 10,    # threshold for "pattern >= pos_limit" -> pattern_identifier_trendbreak = -1
    neg_limit: int = 10     # threshold for "pattern becomes 0 after prev <= -neg_limit" -> +1
) -> pd.DataFrame:
    """
    Pure-Python replacement (no Excel).
    SIGN rules (per row, vs previous row of SAME script):
      sign = +1 if:
        ((close > close_prev) and (low > low_prev or high > high_prev)) OR
        ((close <= close_prev) and (low > low_prev and high > high_prev))
      sign = -1 if:
        ((close < close_prev) and (low < low_prev or high < high_prev)) OR
        ((close >= close_prev) and (low < low_prev and high < high_prev))
      otherwise sign = 0
      (First bar of each script has sign = 0.)

    PATTERN state update (per script, sequential):
      Let K[-1] = 0 at script start. For each bar i:
        If (-5 < K[i-1] < 0) and sign[i] = +1           -> K[i] = 0
        Elif ( 0 < K[i-1] < +5) and sign[i] = -1        -> K[i] = 0
        Elif (|K[i-1]| > 3) and sign[i] opposite to K   -> K[i] = 0
        Else                                            -> K[i] = K[i-1] + sign[i]
      (You can change 5/3 by editing pos_limit/neg_limit and 'hard' 3 below if desired.)

    pattern_identifier_trendbreak:
      +1 if (pattern == 0) and (previous pattern <= -neg_limit)
      -1 if (pattern >= pos_limit)
       0 otherwise
    """

    # -------- Load / prepare input --------
    if df is None:
        if infile is None:
            raise ValueError("Provide either `df` or `infile`.")
        df = pd.read_csv(infile)

    required = ['script', 'Date', 'close_price', 'open_price', 'high_price', 'low_price']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in df: {missing}")

    df = df.copy()
    df = df.sort_values(['script', 'Date'], ascending=True).reset_index(drop=True).drop_duplicates(['script','Date'])


    # -------- Build previous-row columns within each script (vectorized) --------
    grp = df.groupby('script', sort=False, group_keys=False)

    close_prev = grp['close_price'].shift(1)
    low_prev   = grp['low_price'].shift(1)
    high_prev  = grp['high_price'].shift(1)
    same_script_prev = df['script'].eq(df['script'].shift(1))

    # -------- SIGN per your OHLC logic (vectorized) --------
    cond_pos = (
        ((df['close_price'] > close_prev) & ((df['low_price'] > low_prev) | (df['high_price'] > high_prev)))
        |
        ((df['close_price'] <= close_prev) & ((df['low_price'] > low_prev) & (df['high_price'] > high_prev)))
    )

    cond_neg = (
        ((df['close_price'] < close_prev) & ((df['low_price'] < low_prev) | (df['high_price'] < high_prev)))
        |
        ((df['close_price'] >= close_prev) & ((df['low_price'] < low_prev) & (df['high_price'] < high_prev)))
    )

    # sign = +1 / -1 / 0, but force 0 at script boundary
    sign = np.where(~same_script_prev, 0,
                    np.where(cond_pos, 1,
                             np.where(cond_neg, -1, 0)))
    df['sign'] = sign

    # -------- PATTERN state per script (fast NumPy scan per group) --------
    # Reset thresholds:
    # - reset bands use +/-5 (linked to pos_limit/neg_limit)
    # - "large magnitude then opposite" uses 3 (can be changed via 'flip_guard')
    flip_guard = 3

    def _scan_pattern_for_group(idx: np.ndarray) -> np.ndarray:
        # idx: integer positions (in df) of this group's rows
        s = sign[idx]
        K = np.zeros_like(s, dtype=float)
        # sequential scan on raw NumPy (fast; no pandas row loops)
        for i in range(len(idx)):
            if i == 0:
                K[i] = 0.0
                continue
            kprev = K[i-1]
            si    = s[i]

            # Apply reset rules
            if (-neg_limit < kprev < 0) and (si == 1):
                K[i] = 0.0
            elif (0 < kprev < pos_limit) and (si == -1):
                K[i] = 0.0
            elif (abs(kprev) > flip_guard) and (np.sign(kprev) * si < 0):
                K[i] = 0.0
            else:
                K[i] = kprev + si
        return K

    # Apply per script
    pattern = np.empty(len(df), dtype=float)
    for script, sub_idx in df.groupby('script', sort=False).indices.items():
        idx = np.fromiter(sub_idx, dtype=int)
        pattern[idx] = _scan_pattern_for_group(idx)

    df['pattern'] = pattern

    # -------- pattern_identifier_trendbreak (vectorized from pattern) --------
    pattern_prev = grp['pattern'].shift(1)

    df['pattern_identifier_trendbreak'] = np.where((pattern == 0) & (pattern_prev < -pos_limit), 1,
                                            np.where((pattern == 0) & (pattern_prev > pos_limit), -1, 0))


    return df


# ========== NEW CELL ==========

import numpy as np
import pandas as pd

def new_uturn(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # ============================================================
    # 1️⃣ Group setup
    # ============================================================
    grp = df.groupby("script", sort=False, group_keys=False)

    # ============================================================
    # 2️⃣ Uturn_cond1 logic
    # ============================================================
    df["pattern_prev"] = grp["pattern"].shift(1)
    df["pattern_prev2"] = grp["pattern"].shift(2)

    df["Uturn_cond1"] = np.where(
        (df["pattern_prev"] == 0)
        & ((df["pattern_prev2"].abs() >= 4))
        & (df["pattern"].isin([1, -1])),
        1,
        0,
    )

    # ============================================================
    # 3️⃣ Simplified compute_price_change (fully index-safe)
    # ============================================================
    # ============================================================
    # ✅ Simplified, Safe Price Change Computation
    # ============================================================

    def compute_price_change(subdf: pd.DataFrame) -> pd.Series:
        highs = subdf["high_price"].to_numpy()
        lows = subdf["low_price"].to_numpy()
        patterns = subdf["pattern"].to_numpy()

        out = np.full(len(subdf), np.nan)
        for i in range(2, len(subdf)):
            val = patterns[i - 2]
            if np.isnan(val):
                continue
            N = int(abs(val))
            if N < 1:
                continue
            start, end = max(0, i - 2 - N), i - 2
            if start < end:
                high_val = np.nanmax(highs[start:end])
                low_val = np.nanmin(lows[start:end])
                out[i] = high_val - low_val if np.isfinite(high_val) and np.isfinite(low_val) else np.nan
        return pd.Series(out, index=subdf.index)


    # ✅ Handle both single-script and multi-script cases safely
    if "script" in df.columns and df["script"].nunique() > 1:
        price_change_series = (
            df.groupby("script", group_keys=False)
            .apply(compute_price_change)
            .reset_index(level=0, drop=True)
        )
    else:
        # Single script case → compute directly
        price_change_series = compute_price_change(df)

    # Assign safely (aligned to original index)
    df["price_change"] = price_change_series.reindex(df.index)


    # ============================================================
    # 4️⃣ Volatility baseline and significance
    # ============================================================
    # ============================================================
    # ✅ Robust rolling range computation (works for single/multi script)
    # ============================================================

    def rolling_range(subdf: pd.DataFrame, window: int) -> pd.Series:
        return (
            subdf["high_price"].rolling(window).max()
            - subdf["low_price"].rolling(window).min()
        )

    if "script" in df.columns and df["script"].nunique() > 1:
        df["range5"] = (
            df.groupby("script", group_keys=False)
            .apply(lambda x: rolling_range(x, 5))
            .reset_index(level=0, drop=True)
        )
        df["range10"] = (
            df.groupby("script", group_keys=False)
            .apply(lambda x: rolling_range(x, 10))
            .reset_index(level=0, drop=True)
        )
        df["range15"] = (
            df.groupby("script", group_keys=False)
            .apply(lambda x: rolling_range(x, 15))
            .reset_index(level=0, drop=True)
        )
    else:
        # ✅ Single-script case — no need to groupby
        df["range5"] = rolling_range(df, 5)
        df["range10"] = rolling_range(df, 10)
        df["range15"] = rolling_range(df, 15)

    df["avg_range"] = df[["range5", "range10", "range15"]].mean(axis=1)

    df["Price_change_significant"] = np.where(
        df["price_change"] > 0.5 * df["avg_range"], 1, 0
    )

    df["Uturn_cond2"] = np.where(
        (df["pattern_prev"] == 0) & (df["Price_change_significant"] == 1), 1, 0
    )

    # ============================================================
    # 5️⃣ Alert_Uturn logic
    # ============================================================
    cond_alert_pos = (
        (df["Uturn_cond1"] == 1)
        & (df["Uturn_cond2"] == 1)
        & (df["Price_change_significant"] == 1)
        & (df["pattern_prev2"] < 0)
        & (df["pattern"] > 0)
    )
    cond_alert_neg = (
        (df["Uturn_cond1"] == 1)
        & (df["Uturn_cond2"] == 1)
        & (df["Price_change_significant"] == 1)
        & (df["pattern_prev2"] > 0)
        & (df["pattern"] < 0)
    )

    df["Alert_Uturn"] = np.select([cond_alert_pos, cond_alert_neg], [1, -1], default=0)

    # ============================================================
    # 6️⃣ Cleanup temporary columns
    # ============================================================
    df.drop(
        columns=[
            "range5",
            "range10",
            "range15",
            "pattern_prev",
            "pattern_prev2",
        ],
        inplace=True,
        errors="ignore",
    )

    return df


# ========== NEW CELL ==========

import numpy as np
import pandas as pd
from ta.volatility import AverageTrueRange
from ta.trend import ADXIndicator

# =========================
# Minimal helpers
# =========================
def _safe_div(numer, denom, fill=0.0):
    if np.isscalar(denom):
        out = numer / denom if denom != 0 else numer * np.nan
        if isinstance(numer, pd.Series):
            out = pd.Series(out, index=numer.index)
        return out if isinstance(out, pd.Series) else out
    denom_series = denom if isinstance(denom, pd.Series) else pd.Series(denom, index=numer.index)
    out = numer / denom_series.replace(0, np.nan)
    return out.fillna(fill)

def _thr(is_index: bool, tf):
    if tf == 'D':
        if is_index:
            return dict(size_ok=3, size_big=7, cant_miss_size=10, trap_move=0.0030,
                        ok_low=0.0030, ok_high=0.0070, good_thr=0.0070, very_good=0.0150,
                        cant_miss=0.0300, parabolic_size=14, parabolic_move=0.0500)
        else:
            return dict(size_ok=3, size_big=6, cant_miss_size=9, trap_move=0.0060,
                        ok_low=0.0060, ok_high=0.0150, good_thr=0.0150, very_good=0.0350,
                        cant_miss=0.0600, parabolic_size=12, parabolic_move=0.1200)
    if is_index:
        if tf <= 2:
            return dict(size_ok=6, size_big=10, cant_miss_size=15,
                        trap_move=0.0008, ok_low=0.0008, ok_high=0.0012,
                        good_thr=0.0012, very_good=0.0025, cant_miss=0.0045,
                        parabolic_size=20, parabolic_move=0.008)
        else:
            return dict(size_ok=3, size_big=6, cant_miss_size=8,
                        trap_move=0.0008, ok_low=0.0008, ok_high=0.0015,
                        good_thr=0.0015, very_good=0.0030, cant_miss=0.0050,
                        parabolic_size=10, parabolic_move=0.010)
    else:
        if tf <= 2:
            return dict(size_ok=6, size_big=10, cant_miss_size=14,
                        trap_move=0.0015, ok_low=0.0015, ok_high=0.0030,
                        good_thr=0.0030, very_good=0.0060, cant_miss=0.0100,
                        parabolic_size=18, parabolic_move=0.020)
        else:
            return dict(size_ok=3, size_big=6, cant_miss_size=8,
                        trap_move=0.0015, ok_low=0.0015, ok_high=0.0035,
                        good_thr=0.0035, very_good=0.0070, cant_miss=0.0120,
                        parabolic_size=10, parabolic_move=0.025)

def _infer_timeframe(df, timeframe_minutes):
    if timeframe_minutes in (2, 15, 'D'):
        return timeframe_minutes
    if timeframe_minutes == 'auto':
        if 'Date' in df.columns:
            times = df['Date'].dt.time
            if times.notna().all() and all(t == pd.to_datetime("00:00:00").time() for t in times.head(1000)):
                return 'D'
        return 15
    return 15

# =========================
# Combined U-TURN Function (self-contained)
# =========================
def compute_uturn_signals_grouped(
    df=pd.DataFrame(),
    timeframe_minutes=15,
    index_scripts=None,
    u_decline_len=5,
    u_rise_len=5,
    u_consol_window=6,
    u_consol_band_pct=0.007,
    u_breakout_buffer_bps=0.0005,
    adx_threshold=20,
    atr_window=20,
    use_atr_boost=True,
    trend_window=7,
    momentum_window=5
):
    """
    Self-contained U-turn signal detector.
    Automatically derives adx, ema_trend, and slope_momentum internally per script.
    Returns the original df with additional U-turn columns.
    """
    GOOD_ATR_M   = 1.20
    VGOOD_ATR_M  = 1.75
    CANT_ATR_M   = 2.50
    PARAB_ATR_M  = 3.00

    idx_set = set([s.upper() for s in (index_scripts or [])])
    tf_resolved = _infer_timeframe(df, timeframe_minutes)

    def _per_script(x):
        # --- Compute supporting parameters locally ---
        close = x['close_price'].astype(float)
        high  = x['high_price'].astype(float)
        low   = x['low_price'].astype(float)
        open_ = x['open_price'].astype(float) if 'open_price' in x.columns else close.shift(1)

        slope = close.rolling(trend_window).apply(
            lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y)==trend_window else np.nan)
        ema_trend = close - close.ewm(span=trend_window, adjust=False).mean()
        adx = ADXIndicator(high=high, low=low, close=close, window=trend_window).adx()

        slope_momentum = close.rolling(momentum_window).apply(
            lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y)==momentum_window else np.nan)

        scr_name = str(x['script'].iloc[0])
        is_index = scr_name.upper() in idx_set
        TH = _thr(is_index, tf_resolved)

        # --- U-TURN logic (unchanged) ---
        don_high = high.rolling(u_consol_window, min_periods=u_consol_window).max()
        don_low  = low.rolling(u_consol_window, min_periods=u_consol_window).min()
        don_mid  = (don_high + don_low) / 2.0
        band     = _safe_div((don_high - don_low), don_mid.replace(0, np.nan), fill=np.nan)
        is_squeeze = band <= float(u_consol_band_pct)

        avg_price = (open_ + high + low + close) / 4.0
        avg_diff  = avg_price.diff()
        neg_seq = (avg_diff < 0).astype(int)
        pos_seq = (avg_diff > 0).astype(int)
        pre_decline_ok = neg_seq.rolling(u_decline_len, min_periods=u_decline_len).sum().eq(u_decline_len)
        pre_rise_ok    = pos_seq.rolling(u_rise_len,    min_periods=u_rise_len).sum().eq(u_rise_len)

        #breakout_up = (close > (don_high.shift(1) * (1.0 + float(u_breakout_buffer_bps)))) & is_squeeze.shift(1)
        #breakout_dn = (close < (don_low.shift(1)  * (1.0 - float(u_breakout_buffer_bps)))) & is_squeeze.shift(1)


        breakout_up = (close > (don_high * (1.0 + float(u_breakout_buffer_bps)))) & is_squeeze
        breakout_dn = (close < (don_low  * (1.0 - float(u_breakout_buffer_bps)))) & is_squeeze


        adx_ok = adx >= adx_threshold
        mom_up_x = (slope_momentum > 0) & (slope_momentum.shift(1) <= 0) & adx_ok & (ema_trend > 0)
        mom_dn_x = (slope_momentum < 0) & (slope_momentum.shift(1) >= 0) & adx_ok & (ema_trend < 0)

        #start_up_raw = ((breakout_up & pre_decline_ok.shift(1).fillna(False)) | mom_up_x).astype(bool)
        #start_dn_raw = ((breakout_dn & pre_rise_ok.shift(1).fillna(False)) | mom_dn_x).astype(bool)


        start_up_raw = ((breakout_up & pre_decline_ok.fillna(False)) | mom_up_x).astype(bool)
        start_dn_raw = ((breakout_dn & pre_rise_ok.fillna(False)) | mom_dn_x).astype(bool)


        chg = close.diff()
        both = start_up_raw & start_dn_raw
        start_up_raw = start_up_raw.where(~both, chg > 0).astype(bool)
        start_dn_raw = start_dn_raw.where(~both, chg <= 0).astype(bool)

        events_raw = pd.Series(0, index=x.index, dtype=int)
        events_raw = events_raw.mask(start_up_raw, 1).mask(start_dn_raw, -1)

        def _classify_from_events(event_sign: pd.Series):
            event_mask = event_sign.ne(0)
            seg_id = event_mask.cumsum()
            seg_id = seg_id.where(event_mask, np.nan)

            seg_dir = pd.Series(np.nan, index=x.index)
            seg_dir = seg_dir.where(~event_mask, event_sign.astype(float))
            seg_dir = seg_dir.ffill().where(seg_id.notna(), np.nan)

            apex_up_mask = (seg_dir == 1) & (high == high.groupby(seg_id).transform('max')) & seg_id.notna()
            apex_dn_mask = (seg_dir == -1) & (low  == low.groupby(seg_id).transform('min')) & seg_id.notna()
            first_up = apex_up_mask & (apex_up_mask.groupby(seg_id).cumsum() == 1)
            first_dn = apex_dn_mask & (apex_dn_mask.groupby(seg_id).cumsum() == 1)
            cycle_apex_point = pd.Series(0, index=x.index, dtype=int).mask(first_up, -1).mask(first_dn, 1)

            event_avg_price = avg_price.where(event_mask)
            prev_event_avg_price = event_avg_price.ffill().shift(1)
            uturn_pct_change = ((event_avg_price / prev_event_avg_price) - 1).where(event_mask)

            pos = pd.Series(np.arange(len(x)), index=x.index)
            event_pos = pos.where(event_mask)
            prev_event_pos = event_pos.ffill().shift(1)
            cycle_size = (event_pos - prev_event_pos - 1).where(event_mask)

            try:
                atr = AverageTrueRange(high=high, low=low, close=close, window=atr_window).average_true_range()
            except Exception:
                atr = (high - low).rolling(atr_window, min_periods=1).mean()
            atr_pct = _safe_div(atr, avg_price.replace(0, np.nan), fill=np.nan)
            event_atr_pct = atr_pct.where(event_mask).ffill()
            move_vs_atr = _safe_div(uturn_pct_change.abs(), event_atr_pct, fill=np.nan)

            abs_chg = uturn_pct_change.abs()
            cs = cycle_size
            cycle_class = pd.Series(np.nan, index=x.index, dtype="object")

            parabolic = (cs >= TH['parabolic_size']) & (
                (abs_chg >= TH['parabolic_move']) | (use_atr_boost and (move_vs_atr >= PARAB_ATR_M))
            )
            cycle_class = cycle_class.mask(event_mask & parabolic, "parabolic_cycle")

            cant_miss = (cs > TH['cant_miss_size']) & (
                (abs_chg >= TH['cant_miss']) | (use_atr_boost and (move_vs_atr >= CANT_ATR_M))
            )
            cycle_class = cycle_class.mask(event_mask & cant_miss & cycle_class.isna(), "can't_miss_cycle")

            very_good = (cs > TH['size_big']) & (
                (abs_chg >= TH['very_good']) | (use_atr_boost and (move_vs_atr >= VGOOD_ATR_M))
            )
            cycle_class = cycle_class.mask(event_mask & very_good & cycle_class.isna(), "very_good_cycle")

            good = (cs >= TH['size_ok']) & (
                (abs_chg >= TH['good_thr']) | (use_atr_boost and (move_vs_atr >= GOOD_ATR_M))
            )
            cycle_class = cycle_class.mask(event_mask & good & cycle_class.isna(), "good_cycle")

            ok_cond = (cs >= TH['size_ok']) & (cs < TH['size_big']) & (abs_chg > TH['ok_low']) & (abs_chg <= TH['ok_high'])
            cycle_class = cycle_class.mask(event_mask & ok_cond & cycle_class.isna(), "OK_cycle")

            trap = (cs < TH['size_ok']) & (abs_chg <= TH['trap_move'])
            cycle_class = cycle_class.mask(event_mask & trap & cycle_class.isna(), "TRAP")

            grind = (cs > TH['size_big']) & (abs_chg <= TH['ok_high'])
            cycle_class = cycle_class.mask(event_mask & grind & cycle_class.isna(), "grind_cycle")

            cycle_class = cycle_class.mask(event_mask & cycle_class.isna(), "dud_cycle")

            trend_uturn_text = pd.Series("No cycle start", index=x.index)
            trend_uturn_text = trend_uturn_text.mask(event_sign.eq(1),  "Cycle start UP (squeeze breakout / momentum cross)")
            trend_uturn_text = trend_uturn_text.mask(event_sign.eq(-1), "Cycle start DOWN (squeeze breakout / momentum cross)")
            trend_uturn_text = trend_uturn_text.where(~event_mask,
                                                      trend_uturn_text.astype(str) + " | " + cycle_class.astype(str))

            cycle_end = pd.Series(0, index=x.index, dtype=int)
            cycle_end = cycle_end.mask(event_sign.eq(-1), 1).mask(event_sign.eq(1), -1)

            return dict(
                event_sign=event_sign.astype(int),
                event_mask=event_mask,
                cycle_apex_point=cycle_apex_point.astype(int),
                uturn_pct_change=uturn_pct_change.astype(float),
                cycle_size=cycle_size.astype(float),
                cycle_class=cycle_class.astype(str),
                trend_uturn_text=trend_uturn_text.astype(str),
                cycle_end=cycle_end.astype(int)
            )

        res1 = _classify_from_events(events_raw)
        drop_mask = res1['event_mask'] & (res1['cycle_class'].isin(
            ['TRAP', 'dud_cycle', 'trap_cycle', 'DUD', 'DUD_CYCLE']))
        events_pass2 = res1['event_sign'].where(~drop_mask, 0)
        res2 = _classify_from_events(events_pass2)

        uturn_df = pd.DataFrame({
            'trend_uturn': res2['event_sign'],
            'trend_uturn_text': res2['trend_uturn_text'],
            'cycle_start': res2['event_sign'],
            'cycle_end': res2['cycle_end'],
            'cycle_apex_point': res2['cycle_apex_point'],
            'uturn_pct_change': res2['uturn_pct_change'],
            'price_diff_at_cycle_apex': (
                (avg_price.where(res2['event_mask']) - avg_price.where(res2['event_mask']).ffill().shift(1))
            ).astype(float),
            'cycle_size': res2['cycle_size'],
            'cycle_class': res2['cycle_class'],
            'adx': adx,
            'ema_trend': ema_trend,
            'slope_momentum': slope_momentum
        }, index=x.index)

        # Return merged original + U-turn columns
        return x.join(uturn_df)

    # Group apply
    df_out = df.groupby('script', group_keys=False, sort=False).apply(_per_script)
    df_out = df_out.reset_index(drop=True)

    return df_out


# ========== NEW CELL ==========

import pandas as pd
import numpy as np

def calculate_dynamic_sr(df, window=7, pivot_window=5, pivot_min_obs=3, *, fallback_script=None):
    # ✅ Normalize BEFORE anything else
    df = normalize_sr_df(df, fallback_script=fallback_script)

    # Hard guard: Resistance must not run without script
    if df is None or len(df) == 0:
        return df

    if "script" not in df.columns:
        df["script"] = (fallback_script or "UNKNOWN_SCRIPT")

    # ---- your existing logic below (unchanged) ----
    df = df.sort_values(["script", "Date"]).reset_index(drop=True)

    def detect_fractals(group):
        group = group.copy()
        group["fractal_high"] = (
            (group["high_price"] > group["high_price"].shift(1)) &
            (group["high_price"] > group["high_price"].shift(-1))
        )
        group["fractal_low"] = (
            (group["low_price"] < group["low_price"].shift(1)) &
            (group["low_price"] < group["low_price"].shift(-1))
        )
        return group

    def compute_sr(group):
        group = group.copy()
        group["dynamic_resistance"] = np.nan
        group["dynamic_support"] = np.nan

        res_levels = []
        sup_levels = []

        for i in range(len(group)):
            if group.iloc[i]["fractal_high"]:
                res_levels.append(group.iloc[i]["high_price"])
            if group.iloc[i]["fractal_low"]:
                sup_levels.append(group.iloc[i]["low_price"])

            recent_res = res_levels[-window:] if len(res_levels) else []
            recent_sup = sup_levels[-window:] if len(sup_levels) else []

            group.iloc[i, group.columns.get_loc("dynamic_resistance")] = max(recent_res) if recent_res else np.nan
            group.iloc[i, group.columns.get_loc("dynamic_support")] = min(recent_sup) if recent_sup else np.nan

        return group

    df = df.groupby("script", group_keys=False, sort=False).apply(detect_fractals)
    df = df.groupby("script", group_keys=False, sort=False).apply(compute_sr)
    return df

    # ------------------ TRENDLINE COMPUTATION ------------------
    def compute_sr(g):
        n = len(g)
        hp, lp = g['high_price'].to_numpy(), g['low_price'].to_numpy()
        hf_idx = np.where(g['high_fractal'])[0]
        lf_idx = np.where(g['low_fractal'])[0]

        support = np.full(n, np.nan)
        resistance = np.full(n, np.nan)
        high_pts, low_pts = [], []

        for i in range(n):
            # dynamically update fractal points list
            if i in hf_idx:
                high_pts.append((i, hp[i]))
                if len(high_pts) > 3:
                    high_pts.pop(0)
            if i in lf_idx:
                low_pts.append((i, lp[i]))
                if len(low_pts) > 3:
                    low_pts.pop(0)

            # resistance trendline
            if len(high_pts) >= 2:
                (x1, y1), (x2, y2) = high_pts[-2], high_pts[-1]
                if x2 != x1:
                    m_r = (y2 - y1) / (x2 - x1)
                    resistance[i] = y2 + m_r * (i - x2)

            # support trendline
            if len(low_pts) >= 2:
                (x1, y1), (x2, y2) = low_pts[-2], low_pts[-1]
                if x2 != x1:
                    m_s = (y2 - y1) / (x2 - x1)
                    support[i] = y2 + m_s * (i - x2)

        g['Resistance'] = resistance
        g['Support'] = support
        return g

    # ------------------ MAIN PIPELINE ------------------
    df = df.sort_values(['script', 'Date']).reset_index(drop=True)
    df = df.groupby('script', group_keys=False, sort=False).apply(detect_fractals)
    df = df.groupby('script', group_keys=False, sort=False).apply(compute_sr)

    # optional smoothing
    df['Resistance'] = df['Resistance'].fillna(df['high_price']).round(2)
    df['Support'] = df['Support'].fillna(df['low_price']).round(2)

    return df


# ========== NEW CELL ==========


import pandas as pd
import numpy as np

def _dedupe_columns_case_insensitive(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate columns robustly:
    - drops exact duplicates
    - drops case-insensitive duplicates (keeps first occurrence)
    """
    if df is None or df.empty:
        return df

    # Drop exact duplicate column names
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Drop case-insensitive duplicates
    seen = set()
    keep = []
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc not in seen:
            keep.append(c)
            seen.add(lc)

    df = df[keep].copy()
    return df


def _ensure_script_column(df: pd.DataFrame, fallback_script: str = "UNKNOWN_SCRIPT") -> pd.DataFrame:
    """
    Ensures a clean, single 'script' column.
    Handles 'Script' vs 'script', duplicates, and missing.
    """
    if df is None or df.empty:
        # create minimal df if empty to avoid crashes downstream
        if df is None:
            df = pd.DataFrame()
        if "script" not in df.columns:
            df["script"] = fallback_script
        return df

    df = _dedupe_columns_case_insensitive(df)

    cols_lower = {str(c).strip().lower(): c for c in df.columns}

    # If we only have "Script", rename to "script"
    if "script" not in cols_lower and "script" not in df.columns:
        if "script" in cols_lower:
            pass
        elif "script" not in cols_lower and "script" not in df.columns and "script" != "script":
            pass

    # Common case: has 'Script' but no 'script'
    if "script" not in df.columns and "Script" in df.columns:
        df = df.rename(columns={"Script": "script"})

    # If both exist, coalesce then drop Script
    if "script" in df.columns and "Script" in df.columns:
        df["script"] = df["script"].fillna(df["Script"])
        df = df.drop(columns=["Script"], errors="ignore")

    # If still missing, create it
    if "script" not in df.columns:
        df["script"] = fallback_script

    # If script has nulls, fill
    df["script"] = df["script"].fillna(fallback_script).astype(str)

    return df


def _safe_add_missing_columns(df: pd.DataFrame, interval_label: str) -> pd.DataFrame:
    """
    Guarantees all columns required by downstream code exist.
    """
    if df is None or df.empty:
        df = pd.DataFrame()

    df = df.copy()
    required_cols = [
        "pivot_upper", "pivot_lower",
        "Resistance", "Support",
        "screener_implications",
        "user_actions",
        "data_interval"
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    df["data_interval"] = interval_label
    return df


def _safe_fill_sr_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    If SR/pivot cols are missing or NaN, fill them using close_price (safe fallback).
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    if "close_price" not in df.columns:
        # nothing to fill from
        return df

    for col in ["pivot_upper", "pivot_lower", "Resistance", "Support"]:
        if col not in df.columns:
            df[col] = df["close_price"]
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df["close_price"])

    return df






import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def add_pivot_sr(
    df: pd.DataFrame,
    *,
    date_col: str = "Date",
    script_col: str = "script",
    open_col: str = "open_price",
    high_col: str = "high_price",
    low_col: str = "low_price",
    close_col: str = "close_price",
    out_sup_col: str = "pivot_lower",
    out_res_col: str = "pivot_upper",
    window: int = 5,
    min_obs: int = 3
) -> pd.DataFrame:
    """
    Compute per-script Pivot Points on hourly-aggregated data
    and assign Support/Resistance for each 15-min row.
    """

    df = df.copy()

    # ---- SAFETY: fix Script/script + duplicate cols ----
    df = _ensure_script_column(df)
    df = _dedupe_columns_case_insensitive(df)

    
    # ✅ Ensure required columns exist (avoid KeyError surprises)
    if script_col not in df.columns and "Script" in df.columns:
        df = df.rename(columns={"Script": script_col})
    if script_col not in df.columns:
        raise KeyError(f"add_pivot_sr: missing '{script_col}' in df.columns={list(df.columns)}")

    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # drop rows with bad Date to avoid weird groupby/merge outputs
    df = df[df[date_col].notna()].copy()

    # ---- 1. Add hour timestamp ----
    df["Date_hour"] = df[date_col].dt.floor("h")

    # ---- 2. Aggregate to hourly OHLC ----
    hourly = (
        df.groupby([script_col, "Date_hour"], as_index=False)
          .agg({
              open_col: "first",
              high_col: "max",
              low_col: "min",
              close_col: "last"
          })
    )

    # ---- 3. Compute Pivot Points on hourly data ----
    def _calc_hourly_pivots(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("Date_hour").copy()
        g["H_prev"] = g[high_col].rolling(window, min_periods=min_obs).mean().shift(1)
        g["L_prev"] = g[low_col].rolling(window, min_periods=min_obs).mean().shift(1)
        g["C_prev"] = g[close_col].rolling(window, min_periods=min_obs).mean().shift(1)

        g["P"]  = (g["H_prev"] + g["L_prev"] + g["C_prev"]) / 3
        g["R1"] = 2 * g["P"] - g["L_prev"]
        g["S1"] = 2 * g["P"] - g["H_prev"]
        g["R2"] = g["P"] + (g["H_prev"] - g["L_prev"])
        g["S2"] = g["P"] - (g["H_prev"] - g["L_prev"])
        g["R3"] = g["P"] + 2 * (g["H_prev"] - g["L_prev"])
        g["S3"] = g["P"] - 2 * (g["H_prev"] - g["L_prev"])
        return g

    # ✅ Apply pivots per script (Render-safe)
    hourly = hourly.groupby(script_col, group_keys=False).apply(_calc_hourly_pivots)

    # ✅ CRITICAL FIX: ensure script_col is a normal column after groupby.apply()
    if script_col not in hourly.columns:
        hourly = hourly.reset_index()

    # ---- 4. Merge hourly pivots back to 15-min data ----
    hourly_cols = [script_col, "Date_hour", "P", "R1", "R2", "R3", "S1", "S2", "S3"]
    # keep only what we need (and avoid KeyError)
    hourly = hourly[[c for c in hourly_cols if c in hourly.columns]].copy()

    df = df.merge(hourly, on=[script_col, "Date_hour"], how="left")

    # ---- 5. Compute Support/Resistance ----
    def _sr_row(row):
        x = row[close_col]
        pivots = [row.get("S3"), row.get("S2"), row.get("S1"), row.get("P"),
                  row.get("R1"), row.get("R2"), row.get("R3")]
        vals = np.array([v for v in pivots if pd.notna(v)], dtype=float)
        if len(vals) == 0:
            return pd.Series([np.nan, np.nan])
        vals.sort()
        lower = vals[vals <= x]
        higher = vals[vals >= x]
        support = lower[-1] if len(lower) else np.nan
        resistance = higher[0] if len(higher) else np.nan
        return pd.Series([support, resistance])

    df[[out_sup_col, out_res_col]] = df.apply(_sr_row, axis=1)

    # ---- 6. Fallback if not enough history ----
    df[out_sup_col] = df[out_sup_col].fillna(df[low_col])
    df[out_res_col] = df[out_res_col].fillna(df[high_col])

    # ---- 7. Cleanup ----
    df.drop(columns=["P", "R1", "R2", "R3", "S1", "S2", "S3"], inplace=True, errors="ignore")

    return df

    df[[out_sup_col, out_res_col]] = df.apply(_sr_row, axis=1)

    # ---- 6. Fallback if not enough history ----
    df[out_sup_col].fillna(df[low_col], inplace=True)
    df[out_res_col].fillna(df[high_col], inplace=True)

    # ---- 7. Cleanup ----
    drop_cols = ["P", "R1", "R2", "R3", "S1", "S2", "S3"]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    return df


# ========== NEW CELL ==========

import pandas as pd
import numpy as np

def intraday_screener_15min_v3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Advanced intraday screener for 15-min interval data (multi-script).

    Required columns (fixed):
        script, Date, open_price, high_price, low_price, close_price, vol

    Output:
        Original df + the following columns:
          - screener                : str  ("Momentum" | "Reversal" | "NewsDriven" |
                                            "PreBreakout_Zone" | "Volatility_Squeeze" |
                                            "VWAP_Pullback" | "Breakout_Retest" | "")
          - screener_flag           : str  ("screener identified" |
                                            "screener identified no longer valid" | "")
          - screener_implications   : str  (human explanation; now prefixed with side when present)
          - user_actions            : str  (human action guide; now prefixed with side when present)
          - screener_side           : str  ("Bullish" | "Bearish" | "")
    """

    df = df.copy()
    # --- types & sort ---
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values(["script", "Date"], inplace=True)

    # --- Initialize outputs ---
    df["screener"] = ""
    df["screener_flag"] = ""
    df["screener_implications"] = ""
    df["user_actions"] = ""

    # --- Session keys (absolutely critical for intraday logic) ---
    df["Date_daily"] = pd.to_datetime(df["Date"].dt.date)
    df["bar_idx"] = df.groupby(["script", "Date_daily"]).cumcount()

    # --- Base metrics (15-min context) ---
    df["prev_close"] = df.groupby("script")["close_price"].shift(1)

    # Time-of-day aware volume baseline
    df["avg_vol_20"] = df.groupby("script")["vol"].transform(lambda x: x.rolling(20, min_periods=5).mean())
    # Median vol for this bar slot across prior sessions
    df["tod_median_vol"] = (
        df.groupby(["script", "bar_idx"])["vol"]
          .transform(lambda s: s.shift(1).rolling(10, min_periods=5).median())
    )
    df["RVOL_TOD"] = df["vol"] / (df["tod_median_vol"] + 1e-9)

    # intrabar pct change vs prior bar
    df["pct_change"] = ((df["close_price"] - df["prev_close"]) / (df["prev_close"] + 1e-9)) * 100

    # gap vs prior bar (kept intraday; we’ll limit by early session when using it)
    df["Gap%"] = ((df["open_price"] - df["prev_close"]) / (df["prev_close"] + 1e-9)) * 100

    # --- Session VWAP (reset each day) ---
    df["tpv"] = df["close_price"] * df["vol"]
    df["cum_vol_s"] = df.groupby(["script", "Date_daily"])["vol"].cumsum()
    df["cum_tpv_s"] = df.groupby(["script", "Date_daily"])["tpv"].cumsum()
    df["VWAP"] = df["cum_tpv_s"] / (df["cum_vol_s"] + 1e-9)

    # EMAs for context
    df["EMA20"] = df.groupby("script")["close_price"].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    df["EMA50"] = df.groupby("script")["close_price"].transform(lambda x: x.ewm(span=50, adjust=False).mean())
    df["EMA200"] = df.groupby("script")["close_price"].transform(lambda x: x.ewm(span=200, adjust=False).mean())

    # RSI(9) + candle structure for reversals
    def _rsi(series, period=9):
        d = series.diff()
        up = d.clip(lower=0.0)
        dn = -d.clip(upper=0.0)
        avg_up = up.rolling(period, min_periods=period).mean()
        avg_dn = dn.rolling(period, min_periods=period).mean()
        rs = avg_up / (avg_dn + 1e-9)
        return 100 - (100 / (1 + rs))
    df["RSI"] = df.groupby("script")["close_price"].transform(_rsi)

    # Candle anatomy
    body = (df["close_price"] - df["open_price"]).abs()
    trng = (df["high_price"] - df["low_price"]).clip(lower=1e-9)
    upper_shadow = df["high_price"] - df[["open_price", "close_price"]].max(axis=1)
    lower_shadow = df[["open_price", "close_price"]].min(axis=1) - df["low_price"]
    bull_reversal_candle = (lower_shadow >= body) & ((df["close_price"] - df["low_price"]) / trng >= 0.6)
    bear_reversal_candle = (upper_shadow >= body) & ((df["high_price"] - df["close_price"]) / trng >= 0.6)

    # --- Opening Range (first 3 bars) frozen for session ---
    def _orb_high(g):
        # while in first 3 bars: progressive cummax; thereafter: freeze first-3 max
        first3_max = g["high_price"].iloc[:3].max() if len(g) >= 1 else np.nan
        progressive = g["high_price"].cummax()
        return pd.Series(np.where(g["bar_idx"] < 3, progressive, first3_max), index=g.index)

    def _orb_low(g):
        first3_min = g["low_price"].iloc[:3].min() if len(g) >= 1 else np.nan
        progressive = g["low_price"].cummin()
        return pd.Series(np.where(g["bar_idx"] < 3, progressive, first3_min), index=g.index)

    df["ORB_High"] = df.groupby(["script", "Date_daily"], group_keys=False).apply(_orb_high)
    df["ORB_Low"]  = df.groupby(["script", "Date_daily"], group_keys=False).apply(_orb_low)

    # ----------------------------------------------------------------
    #                      Screener Conditions (refined)
    # ----------------------------------------------------------------

    # 1) Opening Range Breakout (both ways)
    df["is_ORB_up"]   = df["close_price"] > df["ORB_High"]
    df["is_ORB_down"] = df["close_price"] < df["ORB_Low"]

    # 2) VWAP Breakout / Retest (momentum piece; both ways, volume-aware)
    df["is_VWAP_up"]   = (df["close_price"] > df["VWAP"]) & (df["vol"] > 1.2 * (df["avg_vol_20"] + 1e-9))
    df["is_VWAP_down"] = (df["close_price"] < df["VWAP"]) & (df["vol"] > 1.2 * (df["avg_vol_20"] + 1e-9))

    # 3) Relative Volume momentum (time-of-day normalized) both ways
    df["is_RVOL_up"]   = (df["RVOL_TOD"] > 1.8) & (df["pct_change"] >  0.6)
    df["is_RVOL_down"] = (df["RVOL_TOD"] > 1.8) & (df["pct_change"] < -0.6)

    # 4) RSI continuation / reversal with candle structure (both ways)
    df["is_RSI"] = (
        (df["RSI"].between(55, 65) & (df["EMA20"] > df["EMA50"]) & bull_reversal_candle) |
        (df["RSI"].between(35, 45) & (df["EMA20"] < df["EMA50"]) & bear_reversal_candle)
    )

    # 5) Gap + ORB Confirmation (limit to early session)
    df["is_GAP"] = (
        (df["Gap%"].abs() > 0.8) &
        (df["bar_idx"] < 4) &
        (
            ((df["Gap%"] > 0) & df["is_ORB_up"]) |
            ((df["Gap%"] < 0) & df["is_ORB_down"])
        )
    )

    # 6) --- PreBreakout_Zone (ACC) — price + volume only, but level-aware & 2-stage ---
    # Volume expansion (last 5 vs previous 5)
    df["vol_sum_last5"] = df.groupby("script")["vol"].transform(lambda x: x.rolling(5).sum())
    df["vol_sum_prev5"] = df.groupby("script")["vol"].transform(lambda x: x.shift(5).rolling(5).sum())
    df["vol_ratio"] = df["vol_sum_last5"] / (df["vol_sum_prev5"] + 1e-9)

    # 5-bar drift (%)
    df["price_change_5"] = (
        (df["close_price"] / (df.groupby("script")["close_price"].shift(5) + 1e-9)) - 1
    ) * 100

    # 5-bar channel and its tightness
    hi5 = df.groupby("script")["high_price"].transform(lambda x: x.rolling(5).max())
    lo5 = df.groupby("script")["low_price"].transform(lambda x: x.rolling(5).min())
    chan_w = (hi5 - lo5)
    df["channel_w_frac"] = chan_w / (df["close_price"] + 1e-9)
    tight = df["channel_w_frac"] <= 0.006  # <= 0.6% on 15-min is a real coil

    # Range contraction vs historical 5-bar range (offset to avoid overlap)
    df["range_last5"] = hi5 - lo5
    df["avg_range_20"] = df.groupby("script")["range_last5"].transform(lambda x: x.shift(5).rolling(20, min_periods=10).mean())
    df["range_ratio"] = df["range_last5"] / (df["avg_range_20"] + 1e-9)

    # Volume steadiness: uniform build-up (std/mean) across last 5 bars
    df["vol_std5"]  = df.groupby("script")["vol"].transform(lambda x: x.rolling(5).std())
    df["vol_mean5"] = df.groupby("script")["vol"].transform(lambda x: x.rolling(5).mean())
    df["vol_steadiness"] = df["vol_std5"] / (df["vol_mean5"] + 1e-9)

    # Local cap/floor context (avoid tagging coils in the middle of nowhere)
    local_cap   = df.groupby("script")["high_price"].transform(lambda x: x.shift(5).rolling(20, min_periods=10).max())
    local_floor = df.groupby("script")["low_price"].transform(lambda x: x.shift(5).rolling(20, min_periods=10).min())
    near_cap_up = (hi5 >= local_cap * 0.9975)
    near_cap_dn = (lo5 <= local_floor * 1.0025)

    # Zone identification (not breakout yet)
    accum = (df["vol_ratio"] > 1.5) & (df["vol_steadiness"] < 0.8) & (df["range_ratio"] < 0.7)
    df["is_ACC_zone_up"] = tight & accum & near_cap_up
    df["is_ACC_zone_dn"] = tight & accum & near_cap_dn

    # Breakout confirmation: wide bar + volume pop + through level
    bar_range = (df["high_price"] - df["low_price"])
    bar_range_ma10 = df.groupby("script")["high_price"].transform(lambda x: x.rolling(10).max()) - \
                     df.groupby("script")["low_price" ].transform(lambda x: x.rolling(10).min())
    wide_bar = bar_range > 0.75 * (bar_range_ma10 + 1e-9)
    vol_pop = df["vol"] > 1.2 * (df["avg_vol_20"] + 1e-9)
    break_up = (df["close_price"] > local_cap) & wide_bar & vol_pop
    break_dn = (df["close_price"] < local_floor) & wide_bar & vol_pop

    df["is_ACC_up"] = df["is_ACC_zone_up"] & break_up
    df["is_ACC_down"] = df["is_ACC_zone_dn"] & break_dn

    # 7) Volatility Squeeze (BB-width percentile; price-only)
    mid = df.groupby("script")["close_price"].transform(lambda s: s.rolling(20, min_periods=15).mean())
    std = df.groupby("script")["close_price"].transform(lambda s: s.rolling(20, min_periods=15).std())
    df["bb_width"] = (4 * std) / (mid + 1e-9)  # total width of ±2σ bands normalized by price
    p20 = df.groupby("script")["bb_width"].transform(lambda s: s.shift(1).rolling(40, min_periods=20).quantile(0.20))
    df["is_SQUEEZE"] = df["bb_width"] <= p20

    # 8) VWAP Pullback (continuation entry, both ways; stronger confirmation)
    near_vwap = df["close_price"].between(df["VWAP"] * 0.9975, df["VWAP"] * 1.0025)
    light_vol = df["vol"] < 0.85 * (df["avg_vol_20"] + 1e-9)
    vol_kick = df["vol"] > df.groupby("script")["vol"].shift(1)

    # Bullish: rejection & reclaim above vwap within bar, uptrend
    rej_up = (df["low_price"] < df["VWAP"]) & (df["close_price"] > df["VWAP"])
    trend_up = df["EMA20"] > df["EMA50"]
    df["is_PULLBACK_up"] = near_vwap & light_vol & rej_up & trend_up & vol_kick

    # Bearish: rejection & reclaim below vwap within bar, downtrend
    rej_dn = (df["high_price"] > df["VWAP"]) & (df["close_price"] < df["VWAP"])
    trend_dn = df["EMA20"] < df["EMA50"]
    df["is_PULLBACK_down"] = near_vwap & light_vol & rej_dn & trend_dn & vol_kick

    # 9) Breakout Retest (structure) both ways; require recent breakout context
    recent_up_break = (
        df.groupby(["script", "Date_daily"])["close_price"]
          .transform(lambda s: s.shift(1).rolling(6, min_periods=1).max())
        > df["ORB_High"]
    )
    recent_dn_break = (
        df.groupby(["script", "Date_daily"])["close_price"]
          .transform(lambda s: s.shift(1).rolling(6, min_periods=1).min())
        < df["ORB_Low"]
    )

    reclaim_up = (df["low_price"] <= df["ORB_High"] * 1.0015) & (df["close_price"] > df["ORB_High"]) & vol_kick
    reclaim_dn = (df["high_price"] >= df["ORB_Low"] * 0.9985) & (df["close_price"] < df["ORB_Low"]) & vol_kick

    df["is_RETEST_up"] = recent_up_break & reclaim_up
    df["is_RETEST_down"] = recent_dn_break & reclaim_dn

    # ----------------------------------------------------------------
    #                     Classification & Messaging
    # ----------------------------------------------------------------
    # Backward-compatible aggregate flags
    df["momentum_flag"] = df["is_ORB_up"] | df["is_ORB_down"] | df["is_VWAP_up"] | df["is_VWAP_down"] | df["is_RVOL_up"] | df["is_RVOL_down"]
    df["reversal_flag"] = df["is_RSI"]
    df["news_flag"]     = df["is_GAP"]
    df["acc_flag"]      = df["is_ACC_up"] | df["is_ACC_down"]
    df["squeeze_flag"]  = df["is_SQUEEZE"]
    df["pullback_flag"] = df["is_PULLBACK_up"] | df["is_PULLBACK_down"]
    df["retest_flag"]   = df["is_RETEST_up"] | df["is_RETEST_down"]

    # (Keep your single-category assignment priority; unchanged names)
    df.loc[df["momentum_flag"], "screener"] = "Momentum"
    df.loc[df["reversal_flag"], "screener"] = "Reversal"
    df.loc[df["news_flag"],     "screener"] = "NewsDriven"
    df.loc[df["acc_flag"],      "screener"] = "PreBreakout_Zone"
    df.loc[df["squeeze_flag"],  "screener"] = "Volatility_Squeeze"
    df.loc[df["pullback_flag"], "screener"] = "VWAP_Pullback"
    df.loc[df["retest_flag"],   "screener"] = "Breakout_Retest"

    # -------- change1: screener_side inference (priority) --------
    df["screener_side"] = ""
    # Momentum side
    df.loc[df["is_ORB_up"] | df["is_VWAP_up"] | df["is_RVOL_up"],   "screener_side"] = "Bullish"
    df.loc[df["is_ORB_down"] | df["is_VWAP_down"] | df["is_RVOL_down"], "screener_side"] = "Bearish"
    # Reversal side (RSI + EMA context)
    df.loc[(df["screener_side"] == "") & (df["RSI"].between(55, 65) & (df["EMA20"] > df["EMA50"])), "screener_side"] = "Bullish"
    df.loc[(df["screener_side"] == "") & (df["RSI"].between(35, 45) & (df["EMA20"] < df["EMA50"])), "screener_side"] = "Bearish"
    # NewsDriven
    df.loc[(df["screener_side"] == "") & (df["Gap%"] > 0) & df["is_ORB_up"],   "screener_side"] = "Bullish"
    df.loc[(df["screener_side"] == "") & (df["Gap%"] < 0) & df["is_ORB_down"], "screener_side"] = "Bearish"
    # ACC
    df.loc[(df["screener_side"] == "") & df["is_ACC_up"],   "screener_side"] = "Bullish"
    df.loc[(df["screener_side"] == "") & df["is_ACC_down"], "screener_side"] = "Bearish"
    # Pullback
    df.loc[(df["screener_side"] == "") & df["is_PULLBACK_up"],   "screener_side"] = "Bullish"
    df.loc[(df["screener_side"] == "") & df["is_PULLBACK_down"], "screener_side"] = "Bearish"
    # Retest
    df.loc[(df["screener_side"] == "") & df["is_RETEST_up"],   "screener_side"] = "Bullish"
    df.loc[(df["screener_side"] == "") & df["is_RETEST_down"], "screener_side"] = "Bearish"
    # -------- end change1 --------

    # Implications & actions (same labels, now enriched with side prefix when present)
    implication_map = {
        "Momentum": "Strong price–volume expansion beyond opening range/VWAP — directional momentum.",
        "Reversal": "RSI and short/medium trend context hint at a potential turn/continuation pivot.",
        "NewsDriven": "Gap with ORB confirmation — event-driven move with broad participation.",
        "PreBreakout_Zone": "Consistent volume build-up with tightening range near a key level — pre-breakout coil.",
        "Volatility_Squeeze": "Volatility compression regime; expansion likely on breakout.",
        "VWAP_Pullback": "Orderly pullback to VWAP on lighter volume within trend — continuation entry zone.",
        "Breakout_Retest": "Breakout followed by shallow retest/reclaim — structure-confirmed continuation.",
    }
    action_map = {
        "Momentum": "Enter with trend; use VWAP/ORB boundary as stop. Trail with swing lows/highs.",
        "Reversal": "Wait for confirming candle near key level; tight stop beyond trigger bar.",
        "NewsDriven": "Trade in gap direction; manage risk tighter due to elevated volatility.",
        "PreBreakout_Zone": "Set alerts above/below the cap/floor; enter on wide-range expansion with volume; stop beyond coil.",
        "Volatility_Squeeze": "Wait for first wide-range breakout candle with RVOL_TOD>1.3, then enter.",
        "VWAP_Pullback": "Enter on reclaim/reject with vol uptick; stop just beyond VWAP.",
        "Breakout_Retest": "Enter on reclaim of the level with vol uptick; stop just past the level.",
    }

    df["screener_implications"] = df["screener"].map(implication_map).fillna("")
    df["user_actions"] = df["screener"].map(action_map).fillna("")

    # -------- change1: side prefixing of texts --------
    df.loc[df["screener"] != "", "screener_implications"] = (
        df["screener_implications"].where(df["screener_side"] == "", df["screener_side"] + " • " + df["screener_implications"])
    )
    df.loc[df["screener"] != "", "user_actions"] = (
        df["user_actions"].where(df["screener_side"] == "", df["screener_side"] + " plan • " + df["user_actions"])
    )
    # -------- end change1 --------

    # --- State management (persist until invalid) ---
    df["prev_screener"] = df.groupby("script")["screener"].shift(1)

    def _alert(row):
        if row["screener"] and row["prev_screener"] != row["screener"]:
            return "screener identified"
        if row["prev_screener"] and not row["screener"]:
            return "screener identified no longer valid"
        return ""
    df["screener_flag"] = df.apply(_alert, axis=1)

    # keep last valid screener active, but clear on invalidation
    df["screener"] = df.groupby("script")["screener"].ffill()
    mask_invalid = df["screener_flag"].eq("screener identified no longer valid")
    df.loc[mask_invalid, ["screener", "screener_implications", "user_actions"]] = ""

    # ----------------------------------------------------------------
    #                       Cleanup temp columns
    # ----------------------------------------------------------------
    drop_cols = [
        # condition flags (helpers)
        "is_ORB_up","is_ORB_down","is_VWAP_up","is_VWAP_down",
        "is_RVOL_up","is_RVOL_down",
        "is_ACC_zone_up","is_ACC_zone_dn","is_ACC_up","is_ACC_down",
        "is_PULLBACK_up","is_PULLBACK_down",
        "is_RETEST_up","is_RETEST_down",
        "is_GAP","is_RSI","is_SQUEEZE",
        # helpers / intermediates
        "prev_screener","avg_vol_20","tod_median_vol","RVOL_TOD","pct_change","Gap%",
        "tpv","cum_vol_s","cum_tpv_s","VWAP",
        "EMA20","EMA50","EMA200","RSI",
        "ORB_High","ORB_Low",
        "channel_w_frac","range_last5","avg_range_20","range_ratio",
        "vol_sum_last5","vol_sum_prev5","vol_ratio","vol_std5","vol_mean5","vol_steadiness",
        "bar_idx","bar_range","Date_daily",
    ]
    # keep screener_side visible; only drop if you truly don't want to expose it
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")

    return df

# ============================================================
# CONSOLIDATION BREAKOUT STRATEGY  (define BEFORE run_strategies)
# ============================================================

import pandas as pd
import numpy as np

def consolidation_breakout_strategy(
    df: pd.DataFrame,
    pct_gain: float = 0.01,
    n_candles: int = 100,
    WIN_CUTOFF: int = 10,
    OUTPUT_CSV: str = "No",
    OUTPUT_CSV_PATH: str = "master_out_2mins.csv"
):
    """Always returns a DataFrame. Safe, no failure."""

    # Safety checks
    if df is None:
        return pd.DataFrame()

    df = df.copy()
    if df.empty:
        df["Alert_Consolidation_Breakout"] = 0
        return df

    df = df.sort_values(["script", "Date"])

    # Rolling breakout zone
    df["Consolidation_High"] = (
        df.groupby("script")["high_price"]
          .transform(lambda x: x.rolling(n_candles, min_periods=20).max())
    )
    df["Consolidation_Low"] = (
        df.groupby("script")["low_price"]
          .transform(lambda x: x.rolling(n_candles, min_periods=20).min())
    )

    # Breakout conditions
    df["UP_Break"] = df["close_price"] >= (df["Consolidation_High"] * (1 + pct_gain))
    df["DN_Break"] = df["close_price"] <= (df["Consolidation_Low"] * (1 - pct_gain))

    df["Alert_Consolidation_Breakout"] = np.where(
        df["UP_Break"], 1,
        np.where(df["DN_Break"], -1, 0)
    )

    # cleanup
    df.drop(columns=["UP_Break", "DN_Break"], inplace=True, errors="ignore")

    if OUTPUT_CSV == "Yes":
        df.to_csv(OUTPUT_CSV_PATH, index=False)

    return df

import pandas as pd
import numpy as np

def build_waterfall_alerts(
    df: pd.DataFrame,
    lookback_start: int = 100,
    min_window_size: int = 3
):
    """
    Safe, guaranteed-return version of Waterfall Alerts.
    Always returns a DataFrame and never fails.
    """

    if df is None:
        return pd.DataFrame()

    df = df.copy()

    if df.empty:
        df["Waterfall_pattern_alert"] = 0
        return df

    df = df.sort_values(["script", "Date"])

    # Price change %
    df["pct_change"] = df.groupby("script")["close_price"].pct_change() * 100

    # Rolling sum of down moves
    df["Down_Move"] = np.where(df["pct_change"] < 0, df["pct_change"], 0)

    df["Down_Sum"] = (
        df.groupby("script")["Down_Move"]
        .transform(lambda x: x.rolling(min_window_size, min_periods=1).sum())
    )

    # Trigger a Waterfall alert
    df["Waterfall_pattern_alert"] = np.where(
        df["Down_Sum"] <= -3,   # example threshold
        -1,
        0
    )

    # Cleanup
    df.drop(columns=["pct_change", "Down_Move"], inplace=True, errors="ignore")

    return df


import pandas as pd
import numpy as np

def generate_trap_alerts(df: pd.DataFrame):
    """
    Safe version of trap alert generator.
    Always returns a DataFrame.
    """

    if df is None:
        return pd.DataFrame()

    df = df.copy()

    if df.empty:
        df["Trap_Alert_POSITION"] = 0
        return df

    df = df.sort_values(["script", "Date"])

    # Identify trap conditions
    df["Trap_Alert_POSITION"] = 0

    # Bullish trap
    df.loc[
        (df["close_price"] > df["open_price"]) &
        (df["low_price"] < df["close_price"] * 0.995),
        "Trap_Alert_POSITION"
    ] = 1

    # Bearish trap
    df.loc[
        (df["close_price"] < df["open_price"]) &
        (df["high_price"] > df["close_price"] * 1.005),
        "Trap_Alert_POSITION"
    ] = -1

    return df



def run_strategies(df1_15min=pd.DataFrame(), df1_2min=pd.DataFrame()):
    # -------------------------
    # SAFETY INIT (prevents unbound vars)
    # -------------------------
    sig_2mins = pd.DataFrame()
    sig_15mins = pd.DataFrame()

    # -------------------------
    # Normalize / De-duplicate columns + ensure script exists
    # -------------------------
    df1_2min = _ensure_script_column(df1_2min)
    df1_15min = _ensure_script_column(df1_15min)

    df1_2min = _dedupe_columns_case_insensitive(df1_2min)
    df1_15min = _dedupe_columns_case_insensitive(df1_15min)

    # -------------------------
    # Ensure Date column type
    # -------------------------
    if "Date" in df1_2min.columns:
        df1_2min["Date"] = pd.to_datetime(df1_2min["Date"], errors="coerce")
    if "Date" in df1_15min.columns:
        df1_15min["Date"] = pd.to_datetime(df1_15min["Date"], errors="coerce")

    # Identify script name for logs (avoid script_name undefined)
    script_name = "UNKNOWN_SCRIPT"
    try:
        if "script" in df1_2min.columns and len(df1_2min) > 0:
            script_name = str(df1_2min["script"].iloc[-1])
        elif "script" in df1_15min.columns and len(df1_15min) > 0:
            script_name = str(df1_15min["script"].iloc[-1])
    except Exception:
        script_name = "UNKNOWN_SCRIPT"

    # -------------------------
    # 2 MIN STRATEGIES
    # -------------------------
    print("FOR 2 MINS DATA")

    print("F1: trend pattern break function is started..")
    df1 = trend_reversal_pattern2(df=df1_2min)
    df1 = _ensure_script_column(df1, fallback_script=script_name)
    print("F1: trend pattern break function is completed!!")

    print("F2: U-turn function only for 2 mins is HALTED TEMPORARILY")
    df1 = new_uturn(df=df1)
    df1 = _ensure_script_column(df1, fallback_script=script_name)
    print("F2: U-turn function is completed!!")

    # IMPORTANT: df2 must exist
    df2 = df1.copy()

    print("F3: Consolidation breakout function is started..")
    df2_2min = consolidation_breakout_strategy(
        df=df1_2min,
        n_candles=100,
        WIN_CUTOFF=7,
        OUTPUT_CSV="No",
        OUTPUT_CSV_PATH="master_out_2mins.csv"
    )
    df2_2min = _ensure_script_column(df2_2min, fallback_script=script_name)
    print("F3: Consolidation breakout function is completed!!")

    print("F4: Waterfall alerts function is started..")
    df3 = build_waterfall_alerts(
        df=df2,
        lookback_start=100,
        min_window_size=3
    )
    df3 = _ensure_script_column(df3, fallback_script=script_name)
    print("F4: Waterfall alerts function is completed!!")

    print("F5: trap Alerts function is started..")
    df3 = generate_trap_alerts(df=df2)
    df3 = _ensure_script_column(df3, fallback_script=script_name)
    print("F5: trap Alerts function is completed!!")

    # -------------------------
    # 15 MIN STRATEGIES
    # -------------------------
    print("FOR 15 MINS DATA")

    print("F1: trend pattern break function is started..")
    df1_15 = trend_reversal_pattern2(df=df1_15min)
    df1_15 = _ensure_script_column(df1_15, fallback_script=script_name)
    print("F1: trend pattern break function is completed!!")

    df1_15 = new_uturn(df=df1_15)
    df1_15 = _ensure_script_column(df1_15, fallback_script=script_name)
    print("F2: U-turn function is completed!!")

    print("F3: Consolidation breakout function is started..")
    df2_15 = consolidation_breakout_strategy(
        df=df1_15,
        n_candles=100,
        WIN_CUTOFF=7,
        OUTPUT_CSV="No",
        OUTPUT_CSV_PATH="master_out_2mins.csv"
    )
    df2_15 = _ensure_script_column(df2_15, fallback_script=script_name)
    print("F3: Consolidation breakout function is completed!!")

    print("F4: Waterfall alerts function is started..")
    df2_15 = build_waterfall_alerts(
        df=df2_15,
        lookback_start=25,
        min_window_size=2
    )
    df2_15 = _ensure_script_column(df2_15, fallback_script=script_name)
    print("F4: Waterfall alerts function is completed!!")

    print("F5: trap Alerts function is started..")
    df3_15 = generate_trap_alerts(df=df2_15)
    df3_15 = _ensure_script_column(df3_15, fallback_script=script_name)
    print("F5: trap Alerts function is completed!!")

    # -------------------------
    # Prepare outputs
    # -------------------------
    sig_2mins = df3.drop_duplicates(["script", "Date"]) if (("script" in df3.columns) and ("Date" in df3.columns)) else df3
    sig_15mins = df3_15.drop_duplicates(["script", "Date"]) if (("script" in df3_15.columns) and ("Date" in df3_15.columns)) else df3_15

    sig_2mins = _ensure_script_column(sig_2mins, fallback_script=script_name)
    sig_15mins = _ensure_script_column(sig_15mins, fallback_script=script_name)

    # -------------------------
    # F6 Resistance (HARDENED)
    # -------------------------
    print("F6: Resistance function is started..")
    try:
        sig_2mins = calculate_dynamic_sr(sig_2mins)
    except Exception as e:
        print(f"[WARN] calculate_dynamic_sr failed (2m) for {script_name}: {e}")

    try:
        sig_2mins = add_pivot_sr(sig_2mins, window=100, out_sup_col="pivot_lower", out_res_col="pivot_upper")
    except Exception as e:
        print(f"[WARN] add_pivot_sr failed (2m) for {script_name}: {e}")

    try:
        sig_15mins = calculate_dynamic_sr(sig_15mins)
    except Exception as e:
        print(f"[WARN] calculate_dynamic_sr failed (15m) for {script_name}: {e}")

    try:
        sig_15mins = add_pivot_sr(sig_15mins, window=100, out_sup_col="pivot_lower", out_res_col="pivot_upper")
    except Exception as e:
        print(f"[WARN] add_pivot_sr failed (15m) for {script_name}: {e}")

    # Guarantee required cols exist even if Resistance failed
    sig_2mins = _safe_add_missing_columns(sig_2mins, "2 mins")
    sig_15mins = _safe_add_missing_columns(sig_15mins, "15 mins")

    # Fill SR cols safely (prevents 'pivot_upper' KeyError)
    sig_2mins = _safe_fill_sr_cols(sig_2mins)
    sig_15mins = _safe_fill_sr_cols(sig_15mins)

    print("F6: Resistance function is completed!!")

    # -------------------------
    # F7 Script Zone (should NOT depend on pivot_upper existing)
    # -------------------------
    print("F7: Script Zone function is started..")
    try:
        sig_2mins = intraday_screener_15min_v3(df=sig_2mins)
    except Exception as e:
        print(f"[WARN] Script Zone failed (2m) for {script_name}: {e}")

    try:
        sig_15mins = intraday_screener_15min_v3(df=sig_15mins)
    except Exception as e:
        print(f"[WARN] Script Zone failed (15m) for {script_name}: {e}")

    print("F7: Script Zone function is completed !!")

    # Ensure final required cols again (zone may drop cols)
    sig_2mins = _safe_add_missing_columns(sig_2mins, "2 mins")
    sig_15mins = _safe_add_missing_columns(sig_15mins, "15 mins")
    sig_2mins = _safe_fill_sr_cols(sig_2mins)
    sig_15mins = _safe_fill_sr_cols(sig_15mins)

    # Final numeric cleanup
    clean_numeric_cols = [
        "pivot_upper", "pivot_lower",
        "Resistance", "Support",
        "close_price", "open_price", "high_price", "low_price"
    ]
    for df in [sig_2mins, sig_15mins]:
        for col in clean_numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Replace remaining NaN/None with safe empty string
    sig_2mins = sig_2mins.replace({np.nan: "", None: ""})
    sig_15mins = sig_15mins.replace({np.nan: "", None: ""})

    return sig_2mins, sig_15mins

    




# ========== NEW CELL ==========

import pandas as pd
import numpy as np

def generate_alerts(sig_2mins: pd.DataFrame = pd.DataFrame(),
                    sig_15mins: pd.DataFrame = pd.DataFrame()):
    # ---------- helpers ----------
    def _add_forward_metrics(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            df["next_10_high"] = np.nan
            df["next_10_low"]  = np.nan
            df["win_buy"] = 0
            df["win_sell"] = 0
            return df

        # Ensure proper ordering within each script
        df = df.sort_values(["script", "Date"]).copy()

        # Average price of THIS candle, then shift(-1) to align "next candle's avg" to current row
        avg_price = (df["open_price"] + df["high_price"] + df["low_price"] + df["close_price"]) / 4.0
        purchase_price = avg_price.groupby(df["script"]).shift(-1)

        # Forward-looking max of next 10 highs (exclude current)
        next_10_high = (
            df.groupby("script", group_keys=False)["high_price"]
              .apply(lambda s: s.iloc[::-1].shift(1).rolling(10, min_periods=1).max().iloc[::-1])
        )
        # Forward-looking min of next 10 lows (exclude current)
        next_10_low = (
            df.groupby("script", group_keys=False)["low_price"]
              .apply(lambda s: s.iloc[::-1].shift(1).rolling(10, min_periods=1).min().iloc[::-1])
        )

        df["next_10_high"] = next_10_high
        df["next_10_low"]  = next_10_low

        # Win if next_10_high >= 1.005 * purchase_price; if purchase_price is NaN (no next bar), mark 0
        df["win_buy"] = np.where(
            purchase_price.notna() & df["next_10_high"].notna() &
            (df["next_10_high"] >= purchase_price * 1.005),
            1, 0
        ).astype(int)

        # Win-sell if next_10_low <= 0.995 * purchase_price; if purchase_price is NaN (no next bar), mark 0
        df["win_sell"] = np.where(
            purchase_price.notna() & df["next_10_low"].notna() &
            (df["next_10_low"] <= purchase_price * 0.995),
            1, 0
        ).astype(int)

        return df

    # ---------- compute forward metrics BEFORE subsetting to today ----------
    # (Input is continuous time series per script)
    if "script" not in sig_2mins.columns:
        sig_2mins = sig_2mins.rename(columns={"Script": "script"})
    if "script" not in sig_15mins.columns:
        sig_15mins = sig_15mins.rename(columns={"Script": "script"})

    sig_2mins = _add_forward_metrics(sig_2mins)
    sig_15mins = _add_forward_metrics(sig_15mins)

    # ---------- derive "today" subset ----------
    sig_2mins["Date_daily"] = pd.to_datetime(pd.to_datetime(sig_2mins["Date"]).dt.date)
    sig_15mins["Date_daily"] = pd.to_datetime(pd.to_datetime(sig_15mins["Date"]).dt.date)

    sig_2mins_today = sig_2mins[sig_2mins.Date_daily >= sig_2mins.Date_daily.max()]
    sig_15mins_today = sig_15mins[sig_15mins.Date_daily >= sig_15mins.Date_daily.max()]

    # ---------- Read existing alerts ----------

    # ---------- Columns to keep ----------
    alert_cols = [c for c in sig_2mins_today.columns if "alert" in c.lower()]
    req_cols = [
        "Date", "script", "close_price", "Resistance", "Support",
        "pivot_upper", "pivot_lower", "screener","screener_side", "screener_implications", "user_actions",
        # forward-test outputs:
        "next_10_high", "next_10_low", "win_buy", "win_sell"
    ]
    otherfixed_alerts_col = ["pattern_identifier_trendbreak"]
    alert_cols.extend(otherfixed_alerts_col)
    req_cols.extend(alert_cols)

    # ---------- Alert mapping ----------
    alert_map = {
        "Alert_Consolidation_Breakout": {1: "Consolidation_breakdown_Bullish", -1: "Consolidation_breakdown_Bearish"},
        "Waterfall_pattern_alert": {1: "Waterfall_Bullish", -1: "Waterfall_Bearish"},
        "Trap_Alert_POSITION": {1: "Trapping_expect Bullish reversal", -1: "Trapping_expect Bearish reversal"},
        "pattern_identifier_trendbreak": {1: "pattern break_expect Bullish reversal", -1: "pattern break_expect Bearish reversal"},
        "Alert_Uturn": {1: "U-turn_expect Bullish reversal", -1: "U-turn_expect Bearish reversal"},
    }

    # ---------- Build today's alerts (2m) ----------
    alerts_today_2mins = sig_2mins_today.loc[:, [c for c in req_cols if c in sig_2mins_today.columns]].copy()
    alerts_today_2mins = alerts_today_2mins[alerts_today_2mins[alert_cols].sum(axis=1) != 0]
    alerts_today_2mins["data_interval"] = "2 mins"
    alerts_today_2mins["Alert_details"] = alerts_today_2mins.apply(
        lambda row: " | ".join(
            alert_map[col][int(row[col])]
            for col in alert_map
            if col in alerts_today_2mins.columns and row[col] in (1, -1)
        ),
        axis=1,
    )

    # ---------- Build today's alerts (15m) ----------
    alerts_today_15mins = sig_15mins_today.loc[:, [c for c in req_cols if c in sig_15mins_today.columns]].copy()
    alerts_today_15mins = alerts_today_15mins[alerts_today_15mins[alert_cols].sum(axis=1) != 0]
    alerts_today_15mins["data_interval"] = "15 mins"
    alerts_today_15mins["Alert_details"] = alerts_today_15mins.apply(
        lambda row: " | ".join(
            alert_map[col][int(row[col])]
            for col in alert_map
            if col in alerts_today_15mins.columns and row[col] in (1, -1)
        ),
        axis=1,
    )

    # ---------- Combine & dedupe ----------
    alerts_today = (
        pd.concat([alerts_today_2mins, alerts_today_15mins], ignore_index=True)
          .drop_duplicates(["Date", "script"])
          .sort_values(["Date", "script"], ascending=False)
    )

    # ---------- Add proximity flags ----------
    alerts_today["Price_Closer_Support"] = (
        ((alerts_today["Resistance"] - alerts_today["close_price"]) / alerts_today["close_price"] >= 0.007)
        & ((alerts_today["close_price"] - alerts_today["Support"]).abs() / alerts_today["Support"] <= 0.003)
    ).astype(int)

    alerts_today["Price_Closer_Resistance"] = (
        ((alerts_today["close_price"] - alerts_today["Support"]) / alerts_today["close_price"] >= 0.007)
        & ((alerts_today["close_price"] - alerts_today["Resistance"]).abs() / alerts_today["Resistance"] <= 0.003)
    ).astype(int)

    return alerts_today



# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# FIX 1 - Add project root to Python path so imports work
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from kiteconnect import KiteConnect

# --------------------------------------------
# FIXED BACKEND PATHS (DO NOT CHANGE)
# --------------------------------------------
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
print("backend_dir_path =", BACKEND_DIR)

INSTRUMENTS_CSV = os.path.join(BACKEND_DIR, "instruments.csv")
#INTERIM_FOLDER = os.path.join(BACKEND_DIR, "data")
#SIG2_PATH = os.path.join(INTERIM_FOLDER, "sig_2mins.csv")
#SIG15_PATH = os.path.join(INTERIM_FOLDER, "sig_15mins.csv")

# --------------------------------------------
# ZERODHA LOGIN
# --------------------------------------------
def get_kite():
    api_key = os.getenv("KITE_API_KEY")
    access_token = os.getenv("KITE_ACCESS_TOKEN")

    if not api_key or not access_token:
        raise Exception("Missing KITE_API_KEY or KITE_ACCESS_TOKEN in .env")

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite

kite = get_kite()

# --------------------------------------------
# REFRESH instruments.csv WHEN MISSING
# --------------------------------------------
if not os.path.exists(INSTRUMENTS_CSV):
    try:
        print("Refreshing instruments.csv ...")
        inst = kite.instruments()
        pd.DataFrame(inst).to_csv(INSTRUMENTS_CSV, index=False)
        print("instruments.csv created successfully!")
    except Exception as e:
        print("Failed to refresh instruments.csv:", e)

# --------------------------------------------
# INSTRUMENT TOKEN LOOKUP
# --------------------------------------------
def load_instruments():
    if not os.path.exists(INSTRUMENTS_CSV):
        raise Exception("instruments.csv not found!")
    df = pd.read_csv(INSTRUMENTS_CSV)
    df["tradingsymbol"] = df["tradingsymbol"].str.upper()
    return df[df["exchange"] == "NSE"]

INSTRUMENTS = load_instruments()



def get_token(symbol: str):
    """Get instrument token for symbol."""
    sym = symbol.upper().replace(".NS", "")
    row = INSTRUMENTS[INSTRUMENTS["tradingsymbol"] == sym]
    if row.empty:
        raise Exception(f"Instrument token not found for {sym}")
    return int(row.iloc[0]["instrument_token"])


# --------------------------------------------
# FETCH HISTORICAL CANDLES
# --------------------------------------------

def fetch_candles(symbol: str, interval: str, days: int = 5):
    """
    interval:
        "2minute", "5minute", "15minute", "60minute", "day"
    """

    token = get_token(symbol)
    print(f"token: {token}")

    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")

    to_date = datetime.now(IST)

    from_date = to_date - timedelta(days=days)

    print(f"Fetching Zerodha candles from {from_date} to {to_date}")

    # -------- FETCH RAW DATA --------
    try:
        data = kite.historical_data(
            instrument_token=token,
            from_date=from_date,
            to_date=to_date,
            interval=interval
        )
    except Exception as e:
        print("❌ Zerodha historical_data error:", e)
        return pd.DataFrame()

    # Zerodha returns list — check length first!
    if not data or len(data) == 0:
        print("❌ No candle data received!")
        return pd.DataFrame()

    print(f"Candles received: {len(data)}")

    # -------- CONVERT TO DATAFRAME --------
    df = pd.DataFrame(data)

    # Guard against missing fields
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            print("❌ Missing candle column:", col)
            return pd.DataFrame()

    # Rename into strategy engine format
    df.rename(columns={
        "date": "Date",
        "open": "open_price",
        "high": "high_price",
        "low": "low_price",
        "close": "close_price",
        "volume": "vol"
    }, inplace=True)

    # Convert date
    df["Date"] = pd.to_datetime(df["Date"])

    # Convert ALL prices into float safely (NO NaN allowed)
    float_cols = ["open_price", "high_price", "low_price", "close_price", "vol"]

    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop any NaN rows (ABSOLUTELY REQUIRED)
    before = len(df)
    df = df.dropna(subset=float_cols).reset_index(drop=True)
    after = len(df)

    if before != after:
        print(f"⚠️ Removed {before-after} rows containing NaN values")

    df["script"] = symbol.upper()

    print(df.tail(5))

    return df


# --------------------------------------------
# NORMALIZE FOR STRATEGY ENGINE
# --------------------------------------------

def normalize(df: pd.DataFrame):
    df = df.copy()
    required = ["script", "Date", "open_price", "high_price", "low_price", "close_price", "vol"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise Exception(f"Missing columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["script", "Date"]).reset_index(drop=True)
    return df


# --------------------------------------------
# MAIN EXECUTION FUNCTION
# --------------------------------------------

def REAL_TIME_ALERTS_INPUTFILES(symbol: str, CSV="No"):
    symbol = symbol.upper()

    print(f"Fetching candles for {symbol}...")

    df2 = fetch_candles(symbol, "2minute", 5)
    df15 = fetch_candles(symbol, "15minute", 30)


    if df2.empty:
        raise Exception(f"2-min candles empty for {symbol}")

    if df15.empty:
        raise Exception(f"15-min candles empty for {symbol}")

    df2 = normalize(df2)
    df15 = normalize(df15)

    if CSV == "Yes":
        df2.to_csv(SIG2_PATH, index=False)
        df15.to_csv(SIG15_PATH, index=False)

    return df2, df15


def REAL_TIME_ALERTS(symbol_list=[], ALERT_TYPE="chart_alerts", CSV="No"):

    # ✅ DO NOT touch df2/df15 here (they don’t exist yet)

    df2_signal = pd.DataFrame()
    df15_signal = pd.DataFrame()

    data_final_2 = pd.DataFrame()
    data_final_15 = pd.DataFrame()

    alerts_today_master = pd.DataFrame()

    for i in symbol_list:
        try:
            # 1) Fetch candles
            df2, df15 = REAL_TIME_ALERTS_INPUTFILES(symbol=i, CSV="No")

            df2 = _canon_cols(df2)
            df15 = _canon_cols(df15)

            if df2.empty or df15.empty:
                raise Exception(f"Empty candles received for {i}")

            # Hard guarantee keys exist
            for name, d in [("df2", df2), ("df15", df15)]:
                if "Date" not in d.columns or "script" not in d.columns:
                    raise Exception(f"{name} missing Date/script. cols={list(d.columns)}")

            data_final_2 = pd.concat([data_final_2, df2], ignore_index=True)
            data_final_15 = pd.concat([data_final_15, df15], ignore_index=True)

            print(f"Data Collected for Script {i}")

            # 2) Run strategies
            sig_2mins, sig_15mins = run_strategies(df1_15min=df15, df1_2min=df2)

            sig_2mins = _canon_cols(sig_2mins)
            sig_15mins = _canon_cols(sig_15mins)

            for name, d in [("sig_2mins", sig_2mins), ("sig_15mins", sig_15mins)]:
                if d is None or d.empty:
                    raise Exception(f"{name} is empty for {i}")
                if "Date" not in d.columns or "script" not in d.columns:
                    raise Exception(f"{name} missing Date/script. cols={list(d.columns)}")

            # 3) Generate alerts
            alerts_today = generate_alerts(sig_2mins=sig_2mins, sig_15mins=sig_15mins)
            alerts_today = _canon_cols(alerts_today)

            if alerts_today is None:
                alerts_today = pd.DataFrame()

            # keep combined alerts (for whatsapp mode)
            if not alerts_today.empty:
                alerts_today_master = pd.concat([alerts_today_master, alerts_today], ignore_index=True)

            # 4) Merge onto candle DF (for chart mode)
            if not alerts_today.empty:
                drop_common_cols = [c for c in alerts_today.columns if c in df2.columns]
                if "Date" in drop_common_cols: drop_common_cols.remove("Date")
                if "script" in drop_common_cols: drop_common_cols.remove("script")

                df2_signal = df2.merge(
                    alerts_today[alerts_today["data_interval"] == "2 mins"].drop(drop_common_cols, axis=1, errors="ignore"),
                    on=["Date", "script"],
                    how="left",
                ).drop_duplicates(["Date", "script"])

                df15_signal = df15.merge(
                    alerts_today[alerts_today["data_interval"] == "15 mins"].drop(drop_common_cols, axis=1, errors="ignore"),
                    on=["Date", "script"],
                    how="left",
                ).drop_duplicates(["Date", "script"])
            else:
                df2_signal = df2.copy()
                df15_signal = df15.copy()

            # 5) Output CSVs for Chart.jsx to read
            if ALERT_TYPE == "chart_alerts":
                if CSV == "Yes":
                    df2_signal.to_csv(SIG2_PATH, index=False)
                    df15_signal.to_csv(SIG15_PATH, index=False)
                    print("✅ Saved:", SIG2_PATH, SIG15_PATH)

                return df2_signal, df15_signal

        except Exception as e:
            print(f"Error in processing {i} → {e}")
            continue

    # whatsapp mode / fallback
    if ALERT_TYPE == "whatsapp_alerts":
        if alerts_today_master.empty:
            print("No Whatsapp Alerts are generated!!")
            return df2_signal, df15_signal

        drop_common_cols = [c for c in alerts_today_master.columns if c in data_final_2.columns]
        if "Date" in drop_common_cols: drop_common_cols.remove("Date")
        if "script" in drop_common_cols: drop_common_cols.remove("script")

        alerts_2 = alerts_today_master[alerts_today_master["data_interval"] == "2 mins"]
        alerts_15 = alerts_today_master[alerts_today_master["data_interval"] == "15 mins"]

        df2_signal = data_final_2.merge(
            alerts_2.drop(drop_common_cols, axis=1, errors="ignore"),
            on=["Date", "script"],
            how="left",
        ).drop_duplicates(["Date", "script"])

        df15_signal = data_final_15.merge(
            alerts_15.drop(drop_common_cols, axis=1, errors="ignore"),
            on=["Date", "script"],
            how="left",
        ).drop_duplicates(["Date", "script"])

        if CSV == "Yes":
            df2_signal.to_csv(SIG2_PATH, index=False)
            df15_signal.to_csv(SIG15_PATH, index=False)

    return df2_signal, df15_signal




# ========== NEW CELL ==========


# DIRECT EXECUTION ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import sys

    # read symbol passed from backend
    symbol = sys.argv[1] if len(sys.argv) > 1 else None

    if not symbol:
        print("❌ No symbol received from backend")
        sys.exit(0)

    print(f"🚀 Running REAL_TIME_ALERTS for: {symbol}")

    # RUN THE FULL PIPELINE
    REAL_TIME_ALERTS(
        symbol_list=[symbol],
        ALERT_TYPE="chart_alerts",
        CSV="Yes"
    )

    print("✅ REAL_TIME_ALERTS execution complete")
