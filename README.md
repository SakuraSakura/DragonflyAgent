# Dragonfly Agent

### Requirements

* pytorch 0.3.0
* numpy
* scipy
* gym
* msgpack-rpc-python
* tensorboardX

## Prepair Environment

### Download pre-trained PSPNet pytorch model

Download [`pspnet50_ADE20K.pth`](https://drive.google.com/open?id=1lB-ABBLghNvhrZQ2ziAjypmRaMD-oFHw)
and store in `pspnet/models/` directory.

### Download Dragonfly drone simulator
** Only Windows is supported **
Download pre-compiled Windows binary [here](https://github.com/kuanting/dragonfly)
You may want to add a `settings.json` file at `FlyingTest/Config` to specify
the port to use
```json
{
  "ApiServerPort": PORT_TO_USE
}
```

## Training

1. Make sure you have created the following directories:
```sh
cd DragonflyAgent
mkdir checkpoints # for a3c model
mkdir runs # for tensorboard log
```

2. Start multiple Dragonfly simulators on consecutive ports.
For example, to train A3C on 4 agents, you need to start 5 simulators
(one for testing) at port 16660 to 16664.

3. In `settings.json`, specify the ip address and port used by Dragonfly simulator.

4. Run the training scripts
```sh
python main.py --num-process <number of training agent> 
```
- use `--no-segmentation` options to disable semantic segmentation.

## Testing

Please refer to `testing.ipynb`.