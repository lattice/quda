# Rename all *.txt to *.text
for f in *.cu; do 
sudo cp -- "$f" "${f%.cu}.cpp"
done
