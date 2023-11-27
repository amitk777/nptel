# create a shell script to run the backBasics.py python file along with the parameters

# python3 backBasics.py -input_size 784 -size 32,16 -lr 0.01 -epochs 15
# python3 backBasics.py -input_size 784 -size 64,32,16 -lr 0.01 -epochs 15
python3 backBasics.py -input_size 784 -size 128,64,32,16 -lr 0.01 -epochs 15
