# Environment Information
Pytorch Geometric <br />
DGL<br />
Pytorch<br />
Numpy<br />
Sklearn<br />
scipy<br />
matplotlib<br />
pip install -r requirements.txt
# HGDL implementation
unzip trained_model.zip file and put all model state file in the current directory
## data for DBLP and YELP dataset is too large to upload in the github. The dataset is stored in the OneDrive.


Example run for HGDL_KL<br />
python main_HGDL.py --dataset drug --hidden 40 --atten 64 --patience 300 --lr 0.005 --gamma 0.0001 --seed 0 <br />
Example run for HAN_KL<br />
python main_HAN.py --dataset urban --hidden 16 --atten 32 --patience 50 --lr 0.005 --seed 0 <br />
Example run for SeHGNN_KL<br />
 python main_SeHGNN.py --dataset urban --hidden 16 --patience 100 --lr 0.005 --seed 0 <br />

