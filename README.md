# PSPNet

chainer implementation of pspnet

## how to use

### step1.  
install chainer
```
pip install chainer=='1.24.0'
```
  
### step2.  
download the cityscape dataset  
your data directory consists of  

your_path|----gtFine  
　　　　　 　|----leftImg8bit  

### step3.  
run the code for training  
the use of multiple gpus is strongly recommended. pspnet consists of very deep layers.
if you have 3 gpus, run  
```
python main.py --data_dir your_path --gpus 0 1 2
```
you can also set the input size and batchsize by the --size and --batchsize option. large input size lead to high performance, please set to larger as possible (depending on your gpu memory).
  

if you have only single gpu  
```
python main.py --data_dir your_path --gpus 0 0 0
```  
in this case, due to the limitation of gpu memory, the input size must be to smaller.  
but even if input size is smaller than 100, pspnet shows good result.  
  
### step4.  
after training 100 epochs, run the code for testing   
```
python main.py --mode test --data_dir your_path --gpus 0 0 0 --load pspnet 100
```  
test is on single gpu


