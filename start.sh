cat > start.sh << 'EOF'
#!/bin/bash
cd /home/susilovps/pest_prediction_v2
pkill -f uvicorn
sleep 2
nohup ~/.local/bin/uvicorn main:app --host 0.0.0.0 --port 8000 > nohup.out 2>&1 &
echo "Server started. PID: $!"
sleep 10
echo "--- Server Output ---"
cat nohup.out
EOF

chmod +x start.sh