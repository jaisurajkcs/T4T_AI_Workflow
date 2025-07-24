from apscheduler.schedulers.blocking import BlockingScheduler
from main import full_pipeline

scheduler = BlockingScheduler()
scheduler.add_job(full_pipeline, 'interval', days=1)
scheduler.start()
