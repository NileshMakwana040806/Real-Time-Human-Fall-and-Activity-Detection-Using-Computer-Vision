from datetime import datetime

def log_event(event):

    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open("event_log.txt", "a", encoding="utf-8") as file:
        file.write(f"{time} : {event}\n")