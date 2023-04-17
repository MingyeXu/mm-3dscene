

## Dependencies
- Ubuntu: 18.04 or higher
- PyTorch: 1.9.0 
- CUDA: 11.1 
- Hardware: 4GPUs 
- To create conda environment, command as follows:


First you can create an anaconda environment called `mm3dscene`:

```bash
conda create -n mm3dscene python=3.7 -y
conda activate mm3dscene
```

Then install Dependecies

```bash
conda install pytorch=1.9.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia -y
conda install -c anaconda h5py pyyaml -y
conda install -c conda-forge sharedarray tensorboardx -y
```

Next install cuda operations

```bash
cd segmentation/lib/pointops
python3 setup.py install
cd ../../..
```
