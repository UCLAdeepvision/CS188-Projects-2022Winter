# UCLA DLCV Course Project

Project page: https://ucladeepvision.github.io/CS188-Projects-2022Winter/

## Instruction for running this site locally

1. Follow the first 2 steps in [pull-request-instruction](pull-request-instruction.md)

2. Installing Ruby with version at least 3.0.0, check https://www.ruby-lang.org/en/documentation/installation/ for instruction.

3. Installing Bundler and jekyll with
```
gem install --user-install bundler jekyll
bundler install
bundle add webrick
```

4. Run your site with
```
bundle exec jekyll serve
```
You should see an address pop on the terminal (http://127.0.0.1:4000/CS188-Projects-2022Winter/ by default), go to this address with your browser.

## Working on the project

1. Create a folder with your team id under ```./assets/images/your-teamid```, you will use this folder to store all the images in your project.

2. Copy the template at ```./_posts/2021-01-18-team00-instruction-to-post.md``` and rename it with format "year-month-date-yourteamid-projectshortname.md" under ```./_posts/```, for example, **2021-01-27-team01-object-detection.md**

3. Check out the sample post we provide at https://ucladeepvision.github.io/CS188-Projects-2022Winter/ and the source code at https://raw.githubusercontent.com/UCLAdeepvision/CS188-Projects-2022Winter/main/_posts/2021-01-18-team00-instruction-to-post.md as well as basic Markdown syntax at https://www.markdownguide.org/basic-syntax/

4. Start your work in your .md file. You may only edit the .md file you just copied and renamed, and add images to ```./assets/images/your-teamid```. Please do NOT change any other files in this repo.

Once you save the .md file, jekyll will synchronize the site and you can check the changes on browser.

## Submission
We will use git pull request to manage submissions.

Once you've done, follow steps 3 and 4 in [pull-request-instruction](pull-request-instruction.md) to make a pull request BEFORE the deadline. Please make sure not to modify any file except your .md file and your images folder. We will merge the request after all submissions are received, and you should able to check your work in the project page on next week of each deadline.

## Deadlines
You should make three pull requests before the following deadlines:

*    January 27: Each group should determine the topic and list the 3 most relevant papers and their code repo.
*    February 24: Each group should include technical details and algorithms/code
*    March 14: Finalize blog article, Colab demo, and recorded video


## Team08 Inital Project Proposal

### Group  Members: Chenda Duan, Zhengtong Liu

### Topic: DeepFake Generation (with emphasis on image to image translation using GAN)

### Relevant papers and their git repo:

1. MaskGAN: Towards Diverse and Interactive Facial Image Manipulation<br>
	GitHub Link: https://github.com/switchablenorms/CelebAMask-HQ

2. StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation<br>
    GitHub Link: https://github.com/yunjey/StarGAN

3. StarGAN v2: Diverse Image Synthesis for Multiple Domains<br>
    GitHub Link: https://github.com/clovaai/stargan-v2

4. Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks<br>
    GitHub Link: https://github.com/junyanz/CycleGAN

5. ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks<br>
    GitHub Link: https://github.com/xinntao/ESRGAN

6. Image-to-Image Translation with Conditional Adversarial Networks<br>
    GitHub Link: https://github.com/phillipi/pix2pix

7. DeepFaceLab: Integrated, flexible and extensible face-swapping framework<br>
    GitHub Link: https://github.com/iperov/DeepFaceLab

8. FSGAN: Subject Agnostic Face Swapping and Reenactment<br>
    GitHub Link: https://github.com/YuvalNirkin/fsgan