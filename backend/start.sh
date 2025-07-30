# start script for render deployment

# set python path
export PYTHONPATH="${PYTHONPATH}:/opt/render/project/src/backend"

# start the application
uvicorn main:app --host 0.0.0.0 --port $PORT