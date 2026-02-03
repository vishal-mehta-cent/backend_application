import json
import sys
from Whatsapp_Push_Module import whatsapp_push_execution

def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "UNKNOWN"

    # message prepared directly (not from CSV)
    msg = f"""
Manual WhatsApp Alert

Script: {symbol}
Triggered from Chart Page
Timestamp: {pd.Timestamp.now()}

Action: Manual Notification
"""
    try:
        whatsapp_push_execution([msg])

        print(json.dumps({
            "status": "success",
            "message": "WhatsApp alert sent",
            "symbol": symbol,
            "messages_sent": 1
        }))
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "error": str(e)
        }))


if __name__ == "__main__":
    import pandas as pd
    main()
