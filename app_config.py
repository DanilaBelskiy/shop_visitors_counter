import os

upload_folder = 'uploads'
if not os.path.isdir(upload_folder):
    os.mkdir(upload_folder)

export_folder = 'result'
if not os.path.isdir(export_folder):
    os.mkdir(export_folder)

stats_folder = 'stats'
if not os.path.isdir(stats_folder):
    os.mkdir(stats_folder)
