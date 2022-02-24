---
layout: post
comments: true
title: Enhanced Self-Driving with combination of map and lidar inputs.
author: Zifan Zhou, Justin Cui
date: 2022-01-18
---


> Self-driving is a hot topic for deep vision applications. However Vision-based urban driving is hard. Lots of methods for learning to drive have been proposed in the past several years. In this work, we focus on reproducing "Learning to drive from a world on rails" and trying to solve the drawbacks of the methods such as high pedestrian friction rate. We will also utilize the lidar(which is preinstalled on most cars with some self-driving capability) data to achieve better bench mark results.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction
Our work is a continuation of [Learning to drive from a world on rails](https://dotchen.github.io/world_on_rails/) which uses dynamic programming to learn an agent from past driving logs and then applies it to generate action values. This work is also very closely related to [Learning by Cheating](https://github.com/dotchen/LearningByCheating) which uses a similar two-stage approach to tackle the learning problem. At the same time, we will compare our work with a few others that uses a totally different approach such as [ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst](https://github.com/aidriver/ChauffeurNet) that learns to drive by synthesizing images to deal with the worst case scenario. In our work, we will try to solve the drawbacks of "Learning to drive from a world on rails" which doesn't use the most accurate Lidar data and has a high pedestrian friction rate problem.

## Technical Details

### Overview
In order to acquire a autonomous drive agent, we divide the work into the following 3 steps:
- Train a Kinematic forward model 
- Based on predefined reward values, compute Q values using Bellman Updates
- With the supervision of precomputed Q values, generate the final model based policy

In terms of training data and evaluation, they are all sampled from the CARLA traffic simulator.


### Simulator
In our work, we use the same traffic simulator(CARLA) as many other similar papers. CARLA has been developed from the ground up to support development, training, and validation of autonomous driving systems. In addition to open-source code and protocols, CARLA provides open digital assets (urban layouts, buildings, vehicles) that were created for this purpose and can be used freely. The simulation platform supports flexible specification of sensor suites, environmental conditions, full control of all static and dynamic actors, maps generation and much more.

Below are a few advantages that CARLA brings to us.

|CARLA Simulator| Instance Segmentation|Different Lighting|
|--|--|--|
![Artificial neural network]({{ '/assets/images/team10/image_1.png' | relative_url }}){: style="height: 150px; max-width: 100%;"} | ![Artificial neural network]({{ '/assets/images/team10/image_2.png' | relative_url }}){: style="height: 150px; max-width: 100%;"} | ![Artificial neural network]({{ '/assets/images/team10/image_5.png' | relative_url }}){: style="height: 150px; max-width: 100%;"}

### Forward Kinematic Model
![Artificial neural network]({{ '/assets/images/team10/bicycle.png' | relative_url }})
{: style="float: right; width: 300px;"}

The goal of forward model is to take as inputs the current ego-vehicle state as 2D location, orientation and speed and predicts the next ego-vehicle state, orientation and speed. In this work, a parameterized bicycle model is used as the structural prior. The basic parameters in a bicycle model are shown on the right.

The bicycle model is called the front wheel steering model, as the front wheel orientation can be controlled relative to the heading of the vehicle. Our target is to compute state [x, y, 𝜃, 𝛿], 𝜃 is heading angle, 𝛿 is steering angle. Our inputs are [𝑣, 𝜑], 𝑣 is velocity, 𝜑 is steering rate.

To analyze the kinematics of the bicycle model, we must select a reference point X, Y on the vehicle which can be placed at the center of the rear axle, the center of the front axle, or at the center of gravity or cg.
![Artificial neural network]({{ '/assets/images/team10/bicycle_analysis.png' | relative_url }})
{: style="width: 500px;"}


Below is the implementation of Kinematic bicycle model in PyTorch
```
class EgoModel(nn.Module):
    def __init__(self, dt=1./4):
        super().__init__()
        
        self.dt = dt

        # Kinematic bicycle model
        self.front_wb = nn.Parameter(torch.tensor(1.),requires_grad=True)
        self.rear_wb  = nn.Parameter(torch.tensor(1.),requires_grad=True)

        self.steer_gain  = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.brake_accel = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.throt_accel = nn.Sequential(
            nn.Linear(1,1,bias=False),
        )
        
    def forward(self, locs, yaws, spds, acts):
        
        '''
        only plannar
        '''
        
        steer = acts[...,0:1]
        throt = acts[...,1:2]
        brake = acts[...,2:3].byte()
        
        accel = torch.where(brake, self.brake_accel.expand(*brake.size()), self.throt_accel(throt))
        wheel = self.steer_gain * steer
        
        beta = torch.atan(self.rear_wb/(self.front_wb+self.rear_wb) * torch.tan(wheel))
        
        next_locs = locs + spds * torch.cat([torch.cos(yaws+beta), torch.sin(yaws+beta)],-1) * self.dt
        next_yaws = yaws + spds / self.rear_wb * torch.sin(beta) * self.dt
        next_spds = spds + accel * self.dt
        
        return next_locs, next_yaws, F.relu(next_spds)
```

### Bellman Equation Evaluation
As the vehicle states in autonumous driving is continuous, it's impossible to model each state. Therefore, a discretization method is used to represent the reward of each state. For each time-step t, the value function is represented as a 4D tensor which is discretized into $$N_H * N_W$$ position bins,$$N_v$$ velocity bins and $$N_{\theta}$$ orientation bins. The value function and action value function can be described by the following 2 equitions

$$V(L_{t}^{ego}, \hat{L}_t^{world}) = \underset{a}{max}Q(L_t^{ego}, \hat{L}_t^{world}, a)$$

$$Q(L_t^{ego}, \hat{L}_t^{world}, a_t) = r(L_t^{ego}, \hat{L}_t^{world}, a_t) + \gamma V(T^{ego}(L_t^{ego}, \hat{L}_t^{world}, a), \hat{L}_{t+1}^{world})$$

The action-value function is needed for all ego-vehicle state $$L^{ego}$$, but only recorded world states $$\hat{L}_t^{world}$$. Therefore, it's scufficient to evaluate the action-value function on just recorded world states for all ego-vehicle:
$$\hat{V}_t(L_t^{ego}) = V(L_t^{ego}, \hat{L}_t^{world}), \hat{Q}_t(L_t^{ego}, a_t) = V(L_t^{ego}, \hat{L}_t^{world}, a_t). $$
Furthermore, since the world states are strictly ordered in time, hence the Bellman equations simplifies to 

$$V(L_{t}^{ego}) = \underset{a}{max}Q(L_t^{ego}, a)$$

$$Q(L_t^{ego}, a_t) = r(L_t^{ego}, \hat{L}_t^{world}, a_t) + \gamma V(T^{ego}(L_t^{ego}, a))$$

It's easy to observe that the above equation uses the compact states(position, orientation and velocity) of the vehicle. Therefore, it can be solved using backward induction and dynamic programming.

Below is the main implementation of computing Q values
```
@staticmethod
def compute_table(ref_yaw, device=torch.device('cuda')):

    ref_yaw = torch.tensor(ref_yaw).float().to(device)

    next_locs = []
    next_yaws = []
    next_spds = []

    locs = torch.zeros((ref_yaw.shape)+(2,)).float().to(device)

    action = BellmanUpdater._actions.expand(len(BellmanUpdater._speeds),len(BellmanUpdater._orient),-1,-1).permute(2,0,1,3)
    speed  = BellmanUpdater._speeds.expand(len(BellmanUpdater._actions),len(BellmanUpdater._orient),-1).permute(0,2,1)[...,None]
    orient = BellmanUpdater._orient.expand(len(BellmanUpdater._actions),len(BellmanUpdater._speeds),-1)[...,None]

    delta_locs, next_yaws, next_spds = BellmanUpdater._ego_model(locs, orient+ref_yaw, speed, action)

    # Normalize to grid's units
    # Note: speed is orientation-agnostic
    delta_locs = delta_locs*PIXELS_PER_METER
    delta_yaws = torch.atan2(torch.sin(next_yaws-ref_yaw-BellmanUpdater._orient[0]), torch.cos(next_yaws-ref_yaw-BellmanUpdater._orient[0]))
    delta_yaws = delta_yaws[...,0,0]/(BellmanUpdater._max_orient-BellmanUpdater._min_orient)*BellmanUpdater._num_orient

    next_spds = (next_spds[...,0,0]-BellmanUpdater._min_speeds)/(BellmanUpdater._max_speeds-BellmanUpdater._min_speeds)*BellmanUpdater._num_speeds

    return delta_locs, delta_yaws, next_spds

```

### Model Based Policy
Based on the Q values computed in the above section, we are able to train a model based policy that generates the final vehicle operating commands. The process to generate the policy can be divided into the following 2 parts.

#### Visual Based Action Prediction
The policy network uses a ResNet34 backbone to parse the RGB inputs. And a global average pooling is used to flatten the ResNet Features before concatenating them with the ego-vehicle speed and feeding it into a fully connected network.

|ResNet34| ResBlock|
|--|--|
![ResNet34]({{ '/assets/images/team10/resnet34.png' | relative_url }}) | ![ResNet34]({{ '/assets/images/team10/res_block.png' | relative_url }})

The network produces a categorical distribution over the discretized action space. In CARLA, the agent receives a high-level navigation command $$c_t$$ for each time step. We supervise the visuomotor agent simultaneously on all the high-level commands. 

The code to build a ResNet34 can be implemented as below
```
class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34,self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=2,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        
        self.block2 = nn.Sequential(
            nn.MaxPool2d(1,1),
            ResidualBlock(64,64),
            ResidualBlock(64,64,2)
        )
        
        self.block3 = nn.Sequential(
            ResidualBlock(64,128),
            ResidualBlock(128,128,2)
        )
        
        self.block4 = nn.Sequential(
            ResidualBlock(128,256),
            ResidualBlock(256,256,2)
        )
        self.block5 = nn.Sequential(
            ResidualBlock(256,512),
            ResidualBlock(512,512,2)
        )
        
        self.avgpool = nn.AvgPool2d(2)
        # vowel_diacritic
        self.fc1 = nn.Linear(512,11)
        # grapheme_root
        self.fc2 = nn.Linear(512,168)
        # consonant_diacritic
        self.fc3 = nn.Linear(512,7)
        
    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        return x1,x2,x3
```

And the code to combine vehiche states with ResNet output to predict the vehicle's next state is as below
```
@torch.no_grad()
def policy(self, wide_rgb, narr_rgb, cmd, spd=None):
    
    assert (self.all_speeds and spd is None) or \
           (not self.all_speeds and spd is not None)
    
    wide_embed = self.backbone_wide(self.normalize(wide_rgb/255.))
    if self.two_cam:
        narr_embed = self.backbone_narr(self.normalize(narr_rgb/255.))
        embed = torch.cat([
            wide_embed.mean(dim=[2,3]),
            self.bottleneck_narr(narr_embed.mean(dim=[2,3])),
        ], dim=1)
    else:
        embed = wide_embed.mean(dim=[2,3])
    
    # Action logits
    if self.all_speeds:
        act_output = self.act_head(embed).view(-1,self.num_cmds,self.num_speeds,self.num_steers+self.num_throts+1)
        # Action logits
        steer_logits = act_output[0,cmd,:,:self.num_steers]
        throt_logits = act_output[0,cmd,:,self.num_steers:self.num_steers+self.num_throts]
        brake_logits = act_output[0,cmd,:,-1]
    else:
        act_output = self.act_head(torch.cat([embed, self.spd_encoder(spd[:,None])], dim=1)).view(-1,self.num_cmds,1,self.num_steers+self.num_throts+1)
        
        # Action logits
        steer_logits = act_output[0,cmd,0,:self.num_steers]
        throt_logits = act_output[0,cmd,0,self.num_steers:self.num_steers+self.num_throts]
        brake_logits = act_output[0,cmd,0,-1]

    return steer_logits, throt_logits, brake_logits
```

#### Image Segmentation
Additionally, the agent is asked to predict semantic segmentation as an auxiliary loss. This consistently improves the agent's driving performance, especially when generalizing to new environments. One of the segmented image is shown as below.

![Artificial neural network]({{ '/assets/images/team10/image_2.png' | relative_url }}){: style="width: 500px; max-width: 100%;"}

The code to implement the segmentation head is as below
```
class SegmentationHead(nn.Module):
    def __init__(self, input_channels, num_labels):
        super().__init__()
        
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(input_channels,256,3,2,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256,128,3,2,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128,64,3,2,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64,num_labels,1,1,0),
        )
        
    def forward(self, x):
        return self.upconv(x)
```

### Reward Design
The reward function $$r(L_t^{ego}, L_t^{world}, a_t, c_t)$$ considers ego-vehicle state, world state, action and high-level command, and is computed from the driving log at each timestep. The agent receives a reward of +1 for staying in the target lane at the desired position, orientation and speed, and is smoothly penalized for deviating from the lane down to a value of 0. If the agent is located at a "zero-speed" region(e.g. red light, or close to other traffic participants), it is rewarded for zero velocity regardless of orientation, and penalized otherwise except for red light zones. All "zero speed" rewards are scaled by $$r_{stop} = 0.01$$, in order to avoid agents disregarding the target lane. 

The agents receives a greedy reward of $$r_{brake} = +5$$ if it brakes in the zero-speed zone. To avoid agents chasing braking region, the braking reward cannot be accumulated. All rewards are additive. One interesting observation is that with zero-speed zones and brake rewards, there is no need to explicitly penalize collisions. We compute the action-values over all high-level commands("turn left", "turn right", "go straight", "follow lane", "change left" or "change right") for each timestep, and use multi-branch supervision when distilling the visumotor agent.

## Demo
<iframe width="560" height="315" src="https://www.youtube.com/embed/1kX7bcJ5cY0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


## Reference
Please make sure to cite properly in your work, for example:

[1] Chen, Koltun, et al. "Learning to drive from a world on rails" *Proceedings of the International Conference on Computer Vision*. 2021.

[2] Dian Chen, Brady Zhou, Vladlen Koltun, and Philipp Kr¨ahenb¨uhl. Learning by cheating. In CoRL, 2019.

[3] Mayank Bansal, Alex Krizhevsky, and Abhijit Ogale. Chauffeurnet: Learning to drive by imitating the best and synthesizing the worst. In RSS, 2019.

## Code Repository
[1] [Learning to drive from a world on rails](https://dotchen.github.io/world_on_rails/)

[2] [Learning by Cheating](https://github.com/dotchen/LearningByCheating)

[3] [ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst](https://github.com/aidriver/ChauffeurNet)

[4] [CARLA: An Open Urban Driving Simulator](https://carla.org/)

---
