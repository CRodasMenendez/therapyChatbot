# install python dependencies
pip install --upgrade pip
pip install -r requirements.txt

python -c "import whisper; whisper.load_model('base')"

echo "Build completed successfully!"